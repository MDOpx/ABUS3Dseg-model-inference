import os

default_num_threads = min(os.cpu_count(), int(os.environ['nnUNet_num_proc'])) if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis
# separately (with NN)