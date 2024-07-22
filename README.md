# ABUS3Dseg-model-inference

## Data/Model preparation
### Data preparation
All the inference data should be positioned in `imagesTs` folder
The TDSC dataset is available on the [TDSC Challenge website](https://tdsc-abus2023.grand-challenge.org/)  
### Model preparation
Trained model should be positioned in `nnUNet_trained_models/nnUNet/3d_fullres/Task500_SA2_TMA_MTL/nnUNetTrainerV2_IBA__nnUNetPlansv2.1` folder
## Tutorial
### 0. Docker setting
Set Docker envoronment for inference
`docker run -dit --name prj --gpus all --shm-size 256g --privileged \
-v /code:/workspace \
loopbackkr/pytorch:1.11.0-cuda11.3-cudnn8`

`docker attach prj`

### 1. Package installation
Install required python packages)
`pip install -r requirements.txt`  
Implementation details  
- Python, version 3.8.12
- Pytorch, version 1.11.0.
- NVIDA A100 with 80 GB

### 2. Environment Setting
Run following lines for inference environemnt setting
`export PYTHONPATH=/workspace
export nnUNet_raw_data_base=/workspace/nnUNet_raw_data_base
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export RESULTS_FOLDER=/workspace/nnUNet_trained_models
export nnUNet_num_proc=64`

### 3. Inference
Run inference code with additional options for SA2+TMA+MTL model.
Results will be saved in `results` Folder
`python hybridnet/inference/predict.py -i imagesTs -o results \
-m nnUNet_trained_models/nnUNet/3d_fullres/Task500_SA2_TMA_MTL/nnUNetTrainerV2_IBA__nnUNetPlansv2.1 -f 0 \
--model_arch MultitaskAGIBA_NI --encoder_module NIConv3dBlock \
--attention_module AGUpConvTwinCBAM3D_wSA_small_v3 --cbammode CS --apply_skips '4' \
--depth [0,1,1,1,1,1] --num_heads [2,2,2,2,2,2] --projection [[1],[1],[1],[1],[1],[1]]`
 
