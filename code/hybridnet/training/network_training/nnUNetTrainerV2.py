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
from hybridnet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from hybridnet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
import torch.nn.functional as F
from hybridnet.network_architecture.static_UNet import Static_UNet, Conv3dBlock

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 use_refinement=False, refinement_loop_count=1, refinement_lazy_start_epoch=0, refinement_mode=None, model_arch=None, loss_ratio_seg=0.7, loss_ratio_cls=0.3, loss_ratio_det=0.0,
                 loss_ratio_mtl_cons=0.0, mtl_cons_start_epoch=0, fixed_validation_set=None, max_num_epochs=1000, encoder_module=Conv3dBlock, decoder_module=Conv3dBlock, train_fp_in_seg=False,
                 legacy_model=False, just_unpacking=False, manual_batch_size=None, swa=False, swa_lr=0.05, swa_start=5, cascade_gap=False, lr_update_off=False, optimizer_name="SGD", manual_device=None,
                 cls_classes=None, initial_lr=1e-2, attention_module=None, iba_estimate_loop=None, cbammode=None, apply_skips='0,1,2,3,4', start_iba_ep=0):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16, lr_update_off)
        self.max_num_epochs = max_num_epochs
        self.initial_lr = initial_lr
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.use_refinement = use_refinement
        self.refinement_loop_count = refinement_loop_count
        self.refinement_lazy_start_epoch = refinement_lazy_start_epoch
        self.refinement_mode = refinement_mode
        self.pin_memory = True
        self.model_arch = model_arch
        self.loss_ratio_seg = loss_ratio_seg
        self.loss_ratio_cls = loss_ratio_cls
        self.loss_ratio_det = loss_ratio_det
        self.loss_ratio_mtl_cons = loss_ratio_mtl_cons
        self.mtl_cons_start_epoch = mtl_cons_start_epoch
        self.fixed_validation_set = fixed_validation_set
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
        self.train_fp_in_seg = train_fp_in_seg
        self.legacy_model = legacy_model
        self.just_unpacking = just_unpacking
        self.manual_batch_size = manual_batch_size
        self.swa = swa
        self.swa_lr = swa_lr
        self.swa_start = swa_start
        self.cascade_gap = cascade_gap
        self.optimizer_name = optimizer_name
        self.manual_device = manual_device
        self.cls_classes = cls_classes

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

            # Initialize
            if self.legacy_model:
                self.network = self.model_arch(num_classes=-1, in_channels=self.num_input_channels, encoder_module=self.encoder_module, decoder_module=self.decoder_module)
            else:
                self.network = self.model_arch(num_classes=self.num_classes, in_channels=self.num_input_channels, encoder_module=self.encoder_module, decoder_module=self.decoder_module,
                                              pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes, conv_kernel_sizes=self.net_conv_kernel_sizes,)

            if self.manual_device is not None:
                self.network.to(self.manual_device)
            else:
                self.network.to('cuda')
            
            self.initialize_optimizer_and_scheduler(optimizer=self.optimizer_name)
            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self, optimizer="SGD"):
        assert self.network is not None, "self.initialize_network must be called first"
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.network.parameters(), self.initial_lr,
                weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
            self.lr_scheduler = None
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.2, patience=self.lr_scheduler_patience,
                verbose=True, threshold=self.lr_scheduler_eps, threshold_mode="abs")
        elif optimizer == "AdaBelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(self.network.parameters(), lr=self.initial_lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
        else:
            raise KeyError(f"Unknown optimizer name: {optimizer}")

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True, is_cls_inference=False, is_grad=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision, is_cls_inference=is_cls_inference, is_grad=is_grad)
        self.network.do_ds = ds
        return ret

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
        # Create one-hot encoded target
        for i in range(len(target)): # round values in each supervision outputs
            target[i] = torch.round(target[i])
            # target[i] = torch.stack([(target[i].squeeze() == j).type(torch.float32) for j in range(self.num_classes)], dim=1)

        if self.fp16:
            with autocast():
                output = self.network(data)
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

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fixed_validation_set:
            from hybridnet.utilities.fixed_validation_sets import FIXED_VALIDATION_SETS

            val_keys = FIXED_VALIDATION_SETS[self.fixed_validation_set][self.fold]
            tr_keys = list(self.dataset.keys())
            for v in val_keys:
                tr_keys.remove(v)
            self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
        elif self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
       
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """

        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                if self.epoch > 0 and self.train_loss_MA is not None:  # otherwise self.train_loss_MA is None
                    self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        else:
            if epoch is None:
                ep = self.epoch + 1
            else:
                ep = epoch
            self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        
        self.print_to_log_file("lr is now %s, Optimizer: %s" % (str(self.optimizer.param_groups[0]['lr']), self.optimizer_name))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
