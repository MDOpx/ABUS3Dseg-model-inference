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


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from hybridnet.run.default_configuration import get_default_configuration
from hybridnet.paths import default_plans_identifier
from hybridnet.run.load_pretrained_weights import load_pretrained_weights
from hybridnet.training.cascade_stuff.predict_next_stage import predict_next_stage
from hybridnet.training.network_training.nnUNetTrainer import nnUNetTrainer
from hybridnet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from hybridnet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from hybridnet.utilities.task_name_id_conversion import convert_id_to_task_name
import pdb
import ast
def arg_as_list(s): #string to list
    v=ast.literal_eval(s)
    return v
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument("--disable_validation_inference", required=False, action="store_true",
                        help="If set nnU-Net will not run inference on the validation set. This is useful if you are "
                             "only interested in the test set results and want to save some disk space and time.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument("--manual_setting", required=False, default=False, action="store_true")
    parser.add_argument("--positive_size", required=False, default=1, type=int)
    parser.add_argument("--negative_size", required=False, default=1, type=int)
    parser.add_argument("--use_refinement", required=False, default=False, action="store_true")
    parser.add_argument("--refinement_loop_count", required=False, type=int, default=0)
    parser.add_argument("--refinement_lazy_start_epoch", required=False, type=int, default=0, help="0 means start from 0 epoch")
    parser.add_argument("--refinement_mode", required=False, type=str, default="+")
    parser.add_argument("--model_arch", default=None, type=str)
    parser.add_argument("--loss_ratio_seg", default=0.7, type=float)
    parser.add_argument("--loss_ratio_cls", default=0.3, type=float)
    parser.add_argument("--loss_ratio_det", default=0.0, type=float)
    parser.add_argument("--encoder_module", default=None, type=str)
    parser.add_argument("--decoder_module", default=None, type=str)
    parser.add_argument("--fixed_validation_set", type=str, required=False, default=None)
    parser.add_argument("--max_num_epochs", type=int, required=False, default=1000)
    parser.add_argument("--manual_patch_size", type=str, required=False, default=None)
    parser.add_argument("--train_fp_in_seg", required=False, action='store_true', default=False)
    parser.add_argument("--legacy_model", required=False, action='store_true', default=False)
    parser.add_argument("--just_unpacking", required=False, action='store_true', default=False)
    parser.add_argument("--manual_batch_size", type=int, required=False, default=None)
    parser.add_argument("--swa", required=False, action='store_true', default=False)
    parser.add_argument("--swa_lr", required=False, default=0.05, type=float)
    parser.add_argument("--cascade_gap", required=False, default=False, type=str)
    parser.add_argument("--lr_update_off", required=False, action='store_true', default=False)
    parser.add_argument("--optimizer_name", required=False, default="SGD", type=str)
    parser.add_argument("--cls_classes", required=False, default=None, type=str)
    parser.add_argument("--load_only_encoders", required=False, action='store_true', default=False)
    parser.add_argument("--initial_lr", default=1e-2, type=float)
    parser.add_argument("--attention_module", default=None)
    parser.add_argument("--iba_estimate_loop", default=10000)
    parser.add_argument("--cbammode", default='CS')
    parser.add_argument("--apply_skips", default='0,1,2,3,4')
    parser.add_argument("--start_iba_ep", type=int, default=0)
    parser.add_argument("--depth", type=arg_as_list, default= [2, 2, 2, 2, 2, 2])
    parser.add_argument("--num_heads", type=arg_as_list, default= [3, 3, 3, 3, 3, 3])
    parser.add_argument("--kernel_size_na", default=7, type=int)
    parser.add_argument("--dilations", type=arg_as_list, default= [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1]])
    parser.add_argument("--reduce_size", type=arg_as_list, default= [5, 5, 5, 5, 5, 5])
    parser.add_argument("--projection", type=arg_as_list, default= [[0], [0], [0], [0], [0], [0]])

    args = parser.parse_args()

    if args.manual_patch_size:
        args.manual_patch_size = [int(i) for i in args.manual_patch_size.split('x')]
    if args.cascade_gap:
        args.cascade_gap = [int(i) for i in args.cascade_gap.split(',')]
    if args.cls_classes:
        args.cls_classes = [i for i in args.cls_classes.split(',')]

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    from nnunet.network_architecture.static_UNet import Static_UNet, Conv3dBlock, ResConv3dBlockRev3, NIConv3dBlock
    from nnunet.network_architecture.static_UNet_IBA import Static_UNet_IBA
    from nnunet.network_architecture.multitask_DNA import MultitaskAGIBA_NI
    if args.model_arch == 'Static_UNet':
        model_arch = Static_UNet
    elif args.model_arch == 'Static_UNet_IBA':
        model_arch = Static_UNet_IBA
    elif args.model_arch == 'MultitaskAGIBA_NI':
        model_arch = MultitaskAGIBA_NI
    else:
        raise KeyError
    trainer = trainer_class(plans_file=plans_file, fold=fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision,
                            use_refinement=args.use_refinement,
                            refinement_loop_count=args.refinement_loop_count,
                            refinement_lazy_start_epoch=args.refinement_lazy_start_epoch,
                            refinement_mode=args.refinement_mode,
                            model_arch=model_arch,
                            loss_ratio_seg=args.loss_ratio_seg,
                            loss_ratio_cls=args.loss_ratio_cls,
                            loss_ratio_det=args.loss_ratio_det,
                            max_num_epochs=args.max_num_epochs,
                            fixed_validation_set=args.fixed_validation_set,
                            encoder_module=eval(args.encoder_module) if args.encoder_module else Conv3dBlock,
                            decoder_module=eval(args.decoder_module) if args.decoder_module else Conv3dBlock,
                            train_fp_in_seg=args.train_fp_in_seg,
                            legacy_model=args.legacy_model,
                            just_unpacking=args.just_unpacking,
                            manual_batch_size=args.manual_batch_size,
                            swa=args.swa,
                            swa_lr=args.swa_lr,
                            cascade_gap=args.cascade_gap,
                            lr_update_off=args.lr_update_off,
                            optimizer_name=args.optimizer_name,
                            cls_classes=args.cls_classes,
                            initial_lr=args.initial_lr,
                            attention_module=args.attention_module,
                            iba_estimate_loop=args.iba_estimate_loop,
                            cbammode=args.cbammode,
                            apply_skips=args.apply_skips,
                            start_iba_ep=args.start_iba_ep,
                            depth=args.depth,
                            num_heads=args.num_heads,
                            kernel_size_na=args.kernel_size_na,
                            dilations=args.dilations,
                            reduce_size=args.reduce_size,
                            projection=args.projection
                            )
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    #trainer.initialize(not validation_only, manual_setting=args.manual_setting, positive_size=args.positive_size, negative_size=args.negative_size, manual_patch_size=args.manual_patch_size)
    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, args.pretrained_weights, load_only_encoders=args.load_only_encoders)
            else:
                # new training without pretraine weights, do nothing
                pass

            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)
            trainer.network.current_epoch = 0

        if args.swa:
            import torch
            network_swa = trainer.network_swa.cpu()
            torch.optim.swa_utils.update_bn(trainer.tr_gen, network_swa)
            network_swa = network_swa.cuda()
            
        trainer.network.eval()

        if args.disable_validation_inference:
            print("Validation inference was disabled. Not running inference on validation set.")
        else:
            # predict validation
            trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                            run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                            overwrite=args.val_disable_overwrite)

        if network == '3d_lowres' and not args.disable_next_stage_pred:
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
