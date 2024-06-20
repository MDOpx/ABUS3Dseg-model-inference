import argparse, os
from typing import Iterable
from os.path import basename, join
import torch
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunet.inference.predict import check_input_folder_and_return_caseIDs
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import multiprocessing.pool as mpp


"""
Implementation of Multi-processing
"""
def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

def run_multiproc(func, *args, num_processes=os.cpu_count()):
    args = list(args)
    num_proc = len(args[0])
    for idx, arg in enumerate(args[1:], start=1):
        if not isinstance(arg, Iterable) or isinstance(arg, str) or len(arg) != num_proc:
            args[idx] = [arg] * num_proc

    with mpp.Pool(min(os.cpu_count(), num_processes)) as pool:
        for _ in pool.istarmap(func, zip(*args)):
            ...
        pool.close()
        pool.join()

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input_folder', required=True)
parser.add_argument('-o', "--output_folder", required=True)
parser.add_argument('--model_dir', required=True)
parser.add_argument('-f', '--folds', required=False, default=None, help="`-1` means all folds")
parser.add_argument("--disable_tta", required=False, default=False, action="store_true", help="disable test time data augmentation via mirroring. Speeds up inference by roughly factor 4 (2D) or 8 (3D)")
parser.add_argument("--overwrite", required=False, default=False, action="store_true", help="Set this flag if the target folder contains predictions that you would like to overwrite")
parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
parser.add_argument("--manual_patch_size", type=str, required=False, default=None)
parser.add_argument("--encoder_module", default=None, type=str)
parser.add_argument("--decoder_module", default=None, type=str)
parser.add_argument("--model_arch", default=None, type=str)
parser.add_argument("--legacy_model", default=False, action='store_true', required=False)
parser.add_argument("--disable_postprocessing", default=False, action='store_true', required=False)
parser.add_argument("--pp_force", default=False, action='store_true', required=False)
parser.add_argument("--disable_pp", default=False, action='store_true', required=False)
parser.add_argument("--swa", required=False, action='store_true', default=False)
parser.add_argument("--cascade_gap", required=False, default=False, type=str)
parser.add_argument("--device_ids", required=False, default=None, type=str, help='Comma seperated device IDs from `CUDA_VISIBLE_DEVICES` for infer each data. `-1` means all gpus available in `CUDA_VISIBLE_DEVICES`.')
args = parser.parse_args()

if args.manual_patch_size:
    args.manual_patch_size = [int(i) for i in args.manual_patch_size.split('x')]
if args.cascade_gap:
    args.cascade_gap = [int(i) for i in args.cascade_gap.split(',')]

model_folder_name = join(args.model_dir, "nnUNetTrainerV2__nnUNetPlansv2.1")
os.makedirs(args.output_folder, exist_ok=True)

expected_num_modalities = load_pickle(join(model_folder_name, "plans.pkl"))['num_modalities']
case_ids = check_input_folder_and_return_caseIDs(args.input_folder, expected_num_modalities)
output_files = [join(args.output_folder, i + ".nii.gz") for i in case_ids]
all_files = sorted(glob(join(args.input_folder, '*.nii.gz')))
list_of_lists = [[i for i in all_files if j in i] for j in case_ids]
lowres_segmentations = None
part_id = 0
num_parts = 1
all_in_gpu = False

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    from nnunet.inference.predict import predict_cases
    if args.device_ids:
        device_ids = [int(id) for id in args.device_ids.split(',')]
        if device_ids == [-1]: # Use all available gpus
            device_count = torch.cuda.device_count()
            device_ids = [i for i in range(device_count)]
        device_ids = [f"cuda:{device_id}" for device_id in device_ids]

        predict_cases(
            [model_folder_name] * device_count,
            [list_of_lists[device_id::device_count] for device_id in range(device_count)],
            [output_files[device_id::device_count] for device_id in range(device_count)],
            args.folds,
            True,                     # save_npz
            1,                        # num_threads_preprocessing
            1,                        # num_threads_nifti_save
            lowres_segmentations,     # segs_from_prev_stage
            False,                    # do_tta
            True,                     # mixed_precision
            True,                     # overwrite_existing
            all_in_gpu,               # all_in_gpu
            0.5,                      # step_size
            "model_final_checkpoint", # checkpoint_name
            None,                     # segmentation_export_kwargs
            False,                    # disable_postprocessing
            args.manual_patch_size,   # manual_patch_size
            args.model_arch,          # model_arch
            args.encoder_module,      # encoder_module
            args.decoder_module,      # decoder_module
            args.legacy_model,        # legacy_model
            args.pp_force,            # pp_force
            args.disable_pp,          # disable_pp
            args.cascade_gap,         # cascade_gap
            device_ids,               # manual_device
        )
    else:
        ...