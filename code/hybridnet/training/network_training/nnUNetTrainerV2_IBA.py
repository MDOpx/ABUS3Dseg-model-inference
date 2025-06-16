#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from hybridnet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from hybridnet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from hybridnet.utilities.to_torch import maybe_to_torch, to_cuda
from hybridnet.network_architecture.generic_UNet import Generic_UNet
from hybridnet.network_architecture.initialization import InitWeights_He
from hybridnet.network_architecture.neural_network import SegmentationNetwork
from hybridnet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from hybridnet.training.dataloading.dataset_loading import unpack_dataset
from hybridnet.training.network_training.nnUNetTrainer import nnUNetTrainer
from hybridnet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from hybridnet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from hybridnet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
import torch.nn.functional as F
from hybridnet.network_architecture.static_UNet import Static_UNet, Conv3dBlock
from hybridnet.network_architecture.multitask_DNA import MultitaskAGIBA_NI


class nnUNetTrainerV2_IBA(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 use_refinement=False, refinement_loop_count=1, refinement_lazy_start_epoch=0, refinement_mode=None, model_arch=None, loss_ratio_seg=0.7, loss_ratio_cls=0.3, loss_ratio_det=0.0,
                 loss_ratio_mtl_cons=0.0,mtl_cons_start_epoch=0,fixed_validation_set=None, max_num_epochs=1000, encoder_module=Conv3dBlock, decoder_module=Conv3dBlock, train_fp_in_seg=False,
                 legacy_model=False, just_unpacking=False, manual_batch_size=None, swa=False, swa_lr=0.05, swa_start=5, cascade_gap=False, lr_update_off=False, optimizer_name="SGD", manual_device=None,
                 cls_classes=None, initial_lr=1e-2, attention_module=None, iba_estimate_loop=10000, cbammode='CS', apply_skips='0,1,2,3,4', start_iba_ep=0, 
                 depth = [2, 2, 2, 2, 2, 2], num_heads = [3, 3, 3, 3, 3, 3], kernel_size_na = 7, dilations = [[1,1], [1,1], [1,1], [1,1], [1,1]],
                 reduce_size = [5, 5, 5, 5, 5, 5], projection = [[0], [0], [0], [0], [0], [0]], camlayer=None):
        super().__init__(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                 unpack_data=unpack_data, deterministic=deterministic, fp16=fp16,
                 use_refinement=use_refinement, refinement_loop_count=refinement_loop_count, refinement_lazy_start_epoch=refinement_lazy_start_epoch, refinement_mode=refinement_mode, model_arch=model_arch, loss_ratio_seg=loss_ratio_seg, loss_ratio_cls=loss_ratio_cls, loss_ratio_det=loss_ratio_det,
                 loss_ratio_mtl_cons=loss_ratio_mtl_cons, mtl_cons_start_epoch=mtl_cons_start_epoch,fixed_validation_set=fixed_validation_set, max_num_epochs=max_num_epochs, encoder_module=encoder_module, decoder_module=decoder_module, train_fp_in_seg=train_fp_in_seg,
                 legacy_model=legacy_model, just_unpacking=just_unpacking, manual_batch_size=manual_batch_size, swa=swa, swa_lr=swa_lr, swa_start=swa_start, cascade_gap=cascade_gap, lr_update_off=lr_update_off, optimizer_name=optimizer_name, manual_device=manual_device,
                 cls_classes=cls_classes, initial_lr=initial_lr)
        self.attention_module = attention_module
        self.iba_estimate_loop = int(iba_estimate_loop)
        self.cbammode = cbammode
        self.apply_skips = apply_skips
        self.start_iba_ep = start_iba_ep
        self.depth = depth
        self.num_heads = num_heads
        self.kernel_size_na = kernel_size_na
        self.dilations = dilations
        self.reduce_size = reduce_size
        self.projection = projection
        self.is_grad = camlayer != None

    def initialize(self, training=True, force_load_plans=False, manual_setting=False, positive_size=1, negative_size=1, manual_patch_size=None):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            if manual_patch_size:
                self.plans['plans_per_stage'][self.plans['num_stages']-1]['patch_size'] = manual_patch_size
                print(f"Manual patch size: {manual_patch_size}")
            self.process_plans(self.plans)
            if self.manual_batch_size:
                self.batch_size = self.manual_batch_size

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            if self.model_arch == Static_UNet:
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            elif self.model_arch == MultitaskAGIBA_NI:
                self.seg_loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                self.cls_loss = nn.BCEWithLogitsLoss()
            else:
                raise KeyError
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators(manual_setting, positive_size, negative_size)
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                    if self.just_unpacking:
                        exit(0)
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=[i for i in range(self.data_aug_params['num_threads'])],
                    seeds_val=[i for i in range(self.data_aug_params['num_threads']//2)],
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass
                # # For IBA
                # self.dl_tr, self.dl_val = self.get_basic_generators(manual_setting, positive_size, negative_size)
                # self.tr_gen, self.val_gen = get_moreDA_augmentation(
                #     self.dl_tr, self.dl_val,
                #     self.data_aug_params[
                #         'patch_size_for_spatialtransform'],
                #     self.data_aug_params,
                #     deep_supervision_scales=self.deep_supervision_scales,
                #     pin_memory=self.pin_memory,
                #     use_nondetMultiThreadedAugmenter=False,
                #     seeds_train=[i for i in range(self.data_aug_params['num_threads'])],
                #     seeds_val=[i for i in range(self.data_aug_params['num_threads']//2)],
                # )

            # Load custom network
            if self.legacy_model:
                self.network = self.model_arch(num_classes=-1, in_channels=self.num_input_channels, encoder_module=self.encoder_module, decoder_module=self.decoder_module)
            else:
                if self.model_arch == MultitaskAGIBA_NI:
                    self.network = self.model_arch(num_classes=self.num_classes, in_channels=self.num_input_channels, encoder_module=self.encoder_module, decoder_module=self.decoder_module, 
                                            pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes, conv_kernel_sizes=self.net_conv_kernel_sizes, patch_size=self.patch_size,
                                            attention_module=self.attention_module, cbammode=self.cbammode, apply_skips=self.apply_skips, start_iba_ep=self.start_iba_ep,
                                            depth = self.depth, num_heads = self.num_heads, reduce_size = self.reduce_size, projection = self.projection, is_grad = self.is_grad)
                else:
                    self.network = self.model_arch(num_classes=self.num_classes, in_channels=self.num_input_channels, encoder_module=self.encoder_module, decoder_module=self.decoder_module, attention_module=self.attention_module, cbammode=self.cbammode, apply_skips=self.apply_skips, start_iba_ep=self.start_iba_ep)
            
            if self.manual_device is not None:
                self.network.to(self.manual_device)
            else:
                 self.network.to('cuda')
            self.initialize_optimizer_and_scheduler(optimizer=self.optimizer_name)
            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
        cls_target_batch = []

        # Create one-hot encoded target
        for i in range(len(target)): # round values in each supervision outputs
            target[i] = torch.round(target[i])
            # target[i] = torch.stack([(target[i].squeeze() == j).type(torch.float32) for j in range(self.num_classes)], dim=1)
        if self.fp16:
            if isinstance(self.network, MultitaskAGIBA_NI):
                with autocast():
                    from torch.profiler import profile, record_function, ProfilerActivity
                    cls_output, output = self.network(data)

                    # Create one-hot encoded labels
                    if self.legacy_model:
                        cls_target = [[0.] if torch.unique(target[0][i]).tolist() == [0.] else [1.] for i, _ in enumerate(labels)]
                        cls_target_batch = to_cuda(torch.Tensor(cls_target))
                    else:
                        for batch_idx in range(target[0].shape[0]): # loop for batch-size, target[0] means top-supervision output
                            cls_target = [0] * self.num_classes
                            uni = torch.unique(target[0][batch_idx]).tolist()
                            if uni == [0.]: # Normal
                                cls_target[0] = 1.
                            elif 1. in uni: # Tumor
                                cls_target[1] = 1.
                            elif 2. in uni: # False positive
                                cls_target[2] = 1.
                            else:
                                raise ValueError(f"Unknown target values: {torch.unique(target[0][batch_idx]).tolist()}")
                            cls_target_batch.append(cls_target)
                        cls_target_batch = to_cuda(torch.Tensor(cls_target_batch)) #[2] -> [2, 1] : prediction 이 [2, 1] 형태로 뱉음

                    seg_loss = self.seg_loss(output, target) * self.loss_ratio_seg
                    if self.loss_ratio_cls > 0:
                        cls_loss = self.cls_loss(cls_output if isinstance(self.cls_loss, nn.CrossEntropyLoss) else F.softmax(cls_output, dim=1), cls_target_batch) * self.loss_ratio_cls
                    else:
                        cls_loss = 0
                    if self.mtl_cons_start_epoch <= self.epoch and self.loss_ratio_mtl_cons > 0:
                        mtl_cons_loss = self.mtl_cons_loss(output[0], cls_output) * self.loss_ratio_mtl_cons
                    else:
                        mtl_cons_loss = 0
                    loss = seg_loss + cls_loss + mtl_cons_loss
                    
                    if do_backprop:
                        self.amp_grad_scaler.scale(loss).backward()
                        self.amp_grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                        self.amp_grad_scaler.step(self.optimizer)
                        self.amp_grad_scaler.update()
                        if self.use_refinement and self.refinement_lazy_start_epoch <= self.epoch:
                            self.optimizer.zero_grad()

                    if self.use_refinement and self.refinement_lazy_start_epoch <= self.epoch:
                        for _ in range(self.refinement_loop_count):
                            seg_probability =  F.softmax(output[0].detach(), 1)
                            
                            seg_tumor_probability = torch.stack([seg_probability[0][1],seg_probability[1][1]])
                            seg_tumor_probability = seg_tumor_probability.unsqueeze(1)
                            
                            if self.refinement_mode == '+':
                                new_input = data + seg_tumor_probability
                            elif self.refinement_mode == '-':
                                new_input = data - seg_tumor_probability
                            else:
                                raise KeyError
                            new_input = to_cuda(new_input)
                            
                            cls_output, output = self.network(new_input)
                            
                            seg_loss = self.seg_loss(output, target)
                            cls_loss = self.cls_loss(F.softmax(cls_output, dim=1), cls_target_batch)
                            loss = seg_loss * self.loss_ratio_seg + cls_loss * self.loss_ratio_cls
                            
                            del new_input
                            
                            if do_backprop:
                                self.amp_grad_scaler.scale(loss).backward()
                                self.amp_grad_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                                self.amp_grad_scaler.step(self.optimizer)
                                self.amp_grad_scaler.update()
                                self.optimizer.zero_grad()
                        
                        # Delete garbages late
                        del data

            else:
                with autocast():
                    # Initialize IBA
                    # self.iba._reset_alpha()
                    # self.iba.optimizer = torch.optim.Adam(lr=self.iba.lr, params=[self.iba.alpha])
                    # self.iba._active_neurons = self.iba.estimator.active_neurons(self.iba._active_neurons_threshold).float()
                    # self.iba._std = torch.max(self.iba.estimator.std(), self.iba.min_std*torch.ones_like(self.iba.estimator.std()))

                    output = self.network(data)#, target=target, std=self.iba._std, mean=self.iba._mean)
                    del data
                    loss = self.loss(output, target)
                    
                if do_backprop:
                    self.amp_grad_scaler.scale(loss).backward()
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
                        
        else:
            raise NotImplementedError
            if isinstance(self.network, Multitask):
                cls_output, output = self.network(data)
                del data
                first_label_size = torch.unique(target[0][0]).size()[0] #2 -> tumor, 1-> normal
                second_label_size = torch.unique(target[0][1]).size()[0] #2 -> tumor, 1-> normal
                cls_target_type.append(first_label_size)
                cls_target_type.append(second_label_size)  
                
                for i in cls_target_type:                    
                    if i == 1:
                        cls_target.append(0)
                    elif i == 2:
                        cls_target.append(1)
                
                cls_target = torch.Tensor(cls_target)#list -> tensor
                cls_target = cls_target.unsqueeze(1)#[2] -> [2, 1] : prediction 이 [2, 1] 형태로 뱉음
                cls_target = to_cuda(cls_target)
                
                seg_loss = self.seg_loss(output, target)
                cls_loss = self.cls_loss(cls_output, cls_target)
                loss = seg_loss * self.loss_ratio_seg + cls_loss * self.loss_ratio_cls
                
                cls_target_type.clear()

                if do_backprop:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()
            else:
                output = self.network(data)
                del data
                loss = self.loss(output, target)

                if do_backprop:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()
        
        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return loss.detach().cpu().numpy()
