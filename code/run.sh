## Data preparation
# 

## Tutorial
# 0. Docker setting
docker run -dit --name prj --gpus all --shm-size 256g --privileged \
-v /{folder}:/workspace \
loopbackkr/pytorch:1.11.0-cuda11.3-cudnn8

docker attach prj

# 1. Package installation
pip install -r requirements.txt

# 2. Environment Setting
export PYTHONPATH=/workspace
export nnUNet_raw_data_base=/workspace/nnUNet_raw_data_base
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export RESULTS_FOLDER=/workspace/nnUNet_trained_models
export nnUNet_num_proc=64

# 3. Inference
python nnunet/inference/predict.py -i nnUNet_raw_data_base/nnUNet_raw_data/Task501_/imagesTs -o results \
-m nnUNet_trained_models/nnUNet/3d_fullres/Task500_SA2_TMA_MTL/nnUNetTrainerV2_IBA__nnUNetPlansv2.1 -f 0 \
--model_arch MultitaskAGIBA_NI --encoder_module NIConv3dBlock \
--attention_module AGUpConvTwinCBAM3D_wSA_small_v3 --cbammode CS --apply_skips '4' \
--depth [0,1,1,1,1,1] --num_heads [2,2,2,2,2,2] --projection [[1],[1],[1],[1],[1],[1]]
 
# 4. CAM
python nnunet/inference/predict.py -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_/imagesTs -o results \
-m nnUNet_trained_models/nnUNet/3d_fullres/Task500_SA2_TMA_MTL/nnUNetTrainerV2_IBA__nnUNetPlansv2.1 -f 0 \
--model_arch MultitaskAGIBA_NI --encoder_module NIConv3dBlock \
--attention_module AGUpConvTwinCBAM3D_wSA_small_v3 --cbammode CS --apply_skips '4' \
--depth [0,1,1,1,1,1] --num_heads [2,2,2,2,2,2] --projection [[1],[1],[1],[1],[1],[1]] --cam_layer 'seg_decoder.decoder.4'
