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

import nnunet
import torch
from batchgenerators.utilities.file_and_folder_operations import *
import importlib
import pkgutil
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer


def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None, manual_patch_size=None, model_arch=None, encoder_module=None, decoder_module=None, legacy_model=False, cascade_gap=False, manual_device=None, swa=False, cls_classes=False,
                    attention_module=None, iba_estimate_loop=10000, cbammode=None, apply_skips=None, 
                    depth=[0,0,0,0,0,0], num_heads=[0,0,0,0,0,0], reduce_size=[0,0,0,0,0,0], projection=[[0],[0],[0],[0],[0],[0]],
                    camlayer = None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")

    if tr is None:
        """
        Fabian only. This will trigger searching for trainer classes in other repositories as well
        """
        try:
            import meddec
            search_in = join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class([search_in], name, current_module="meddec.model_training")
        except ImportError:
            pass

    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...
    trainer = tr(*init, model_arch=model_arch, encoder_module=encoder_module, decoder_module=decoder_module, legacy_model=legacy_model, cascade_gap=cascade_gap, manual_device=manual_device, swa=swa, cls_classes=cls_classes,
                attention_module=attention_module, iba_estimate_loop=iba_estimate_loop, cbammode=cbammode, apply_skips=apply_skips, 
                depth=depth, num_heads=num_heads, reduce_size=reduce_size, projection=projection, camlayer=camlayer)


    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if manual_patch_size:
        trainer.plans['plans_per_stage'][trainer.plans['num_stages']-1]['patch_size'] = manual_patch_size
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_best_model_for_inference(folder):
    checkpoint = join(folder, "model_best.model")
    pkl_file = checkpoint + ".pkl"
    return restore_model(pkl_file, checkpoint, False)


def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best", manual_patch_size=None,
        model_arch=None, encoder_module=None, decoder_module=None, legacy_model=False, cascade_gap=False, manual_device=None, swa=False, cls_classes=False,
        attention_module=None, iba_estimate_loop=10000, cbammode=None, apply_skips=None, 
        depth=[0,0,0,0,0,0], num_heads=[0,0,0,0,0,0], reduce_size=[0,0,0,0,0,0], projection=[[0],[0],[0],[0],[0],[0]],
        camlayer = None):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    from nnunet.network_architecture.static_UNet import Conv3dBlock, NIConv3dBlock
    from nnunet.network_architecture.multitask_DNA import MultitaskAGIBA_NI
    #DEL
    #from nnunet.network_architecture.static_UNet import Static_UNet, Conv3dBlock#, Conv3dVBlock, ResConv3dBlock, ResConv3dBlockRev3, Dense3dBlock_CBR, ResConv3dTFBlock, NIConv3dBlock
    # from nnunet.network_architecture.multitask import Multitask
    # from nnunet.network_architecture.multitask_6L import Multitask6L
    # from nnunet.network_architecture.multitask_AG import MultitaskAG
    # from nnunet.network_architecture.multitask_CLS import MultitaskCLS
    # from nnunet.network_architecture.multitask_DET import MultitaskDET
    if model_arch !=None: model_arch = eval(model_arch) 
    if encoder_module !=None: encoder_module = eval(encoder_module)
    else: encoder_module = Conv3dBlock
    if decoder_module !=None: decoder_module = eval(decoder_module) 
    else: decoder_module = Conv3dBlock
    trainer = restore_model(join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision, manual_patch_size=manual_patch_size, model_arch=model_arch,
        encoder_module=encoder_module, decoder_module=decoder_module, legacy_model=legacy_model, cascade_gap=cascade_gap, manual_device=manual_device, swa=swa, cls_classes=cls_classes,
        attention_module=attention_module, iba_estimate_loop=iba_estimate_loop, cbammode=cbammode, apply_skips=apply_skips, 
        depth=depth, num_heads=num_heads, reduce_size=reduce_size, projection=projection, camlayer=camlayer)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params


if __name__ == "__main__":
    pkl = "/home/fabian/PhD/results/nnUNetV2/nnUNetV2_3D_fullres/Task004_Hippocampus/fold0/model_best.model.pkl"
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
