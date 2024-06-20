# Python build-ins
from typing import Tuple, Union, List
# PyTorch
import torch, torch.nn as nn, numpy as np
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU
from torch.cuda.amp import autocast
# nnUNet modules
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from scipy.ndimage.filters import gaussian_filter
from batchgenerators.augmentations.utils import pad_nd_image
import torch.nn.functional as F
#from natten import NeighborhoodAttention3D as NeighborhoodAttention
import pdb
from einops import rearrange
import math


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple=(3, 3, 3), stride0: tuple=(1, 1, 1), stride1: tuple=(1, 1, 1), padding: tuple=(0, 0, 0), bias: bool=True) -> None:
        super().__init__()
        
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size, stride0, padding, bias=bias)
        self.instnorm0 = InstanceNorm3d(out_channels, affine=True)
        self.act0 = LeakyReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size, stride1, padding, bias=bias)
        self.instnorm1 = InstanceNorm3d(out_channels, affine=True)
        self.act1 = LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.instnorm0(x)
        x = self.act0(x)

        x = self.conv1(x)
        x = self.instnorm1(x)
        x = self.act1(x)
        return x

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Static_UNet(nn.Module):
    # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
    # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
    # can be expensive, so it makes sense to save and reuse them.
    _gaussian_3d = _patch_size_for_gaussian_3d = None

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        encoder_module: nn.Module=Conv3dBlock,
        decoder_module: nn.Module=Conv3dBlock,
    ):
        super().__init__()
        self.num_classes = num_classes if not num_classes == -1 else 2

        # For origin nnUNet v1 backward compatibility
        self.do_ds = True
        self.final_nonlin = lambda x: x
        self.deep_supervision = True

        # Encoder phase
        self.encoder = nn.ModuleList([
            encoder_module(in_channels,  32, padding=1),
            encoder_module(         32,  64, padding=1, stride0=2, stride1=1),
            encoder_module(         64, 128, padding=1, stride0=2, stride1=1),
            encoder_module(        128, 256, padding=1, stride0=2, stride1=1),
            encoder_module(        256, 320, padding=1, stride0=2, stride1=1),
        ])
        # Bottle-neck
        self.bottleneck = Conv3dBlock(320, 320, kernel_size=3, padding=1, stride0=(1, 2, 2), stride1=1)

        # Decoder phase
        self.decoder = nn.ModuleList([
            decoder_module(640, 320, padding=1),
            decoder_module(512, 256, padding=1),
            decoder_module(256, 128, padding=1),
            decoder_module(128,  64, padding=1),
            decoder_module( 64,  32, padding=1),
        ])
        
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            nn.ConvTranspose3d(320, 256, kernel_size=2,         stride=2,         bias=False),
            nn.ConvTranspose3d(256, 128, kernel_size=2,         stride=2,         bias=False),
            nn.ConvTranspose3d(128,  64, kernel_size=2,         stride=2,         bias=False),
            nn.ConvTranspose3d( 64,  32, kernel_size=2,         stride=2,         bias=False),
        ])
        
        # For deep supervision
        self.deepsupervision = nn.ModuleList([
            nn.Conv3d(320, 2, kernel_size=1, stride=1, bias=False),
            nn.Conv3d(256, 2, kernel_size=1, stride=1, bias=False),
            nn.Conv3d(128, 2, kernel_size=1, stride=1, bias=False),
            nn.Conv3d( 64, 2, kernel_size=1, stride=1, bias=False),
            nn.Conv3d( 32, 2, kernel_size=1, stride=1, bias=False),
        ])

        self.apply(InitWeights_He)
        self.upscale_logits_ops = [lambda x: x for _ in range(4)]
    
    def forward(self, x):
        skips = []
        seg_outputs = []

        x_enc0 = self.encoder[0](x)
        skips.append(x_enc0)
        x_enc1 = self.encoder[1](x_enc0)
        skips.append(x_enc1)
        x_enc2 = self.encoder[2](x_enc1)
        skips.append(x_enc2)
        x_enc3 = self.encoder[3](x_enc2)
        skips.append(x_enc3)
        x_enc4 = self.encoder[4](x_enc3)
        skips.append(x_enc4)

        x_enc5 = self.bottleneck(x_enc4)

        x_dec0 = self.upsamplers[0](x_enc5)
        x_dec0 = torch.cat((x_dec0, skips[4]), dim=1)
        x_dec0 = self.decoder[0](x_dec0)
        seg_outputs.append(self.final_nonlin(self.deepsupervision[0](x_dec0)))

        x_dec1 = self.upsamplers[1](x_dec0)
        x_dec1 = torch.cat((x_dec1, skips[3]), dim=1)
        x_dec1 = self.decoder[1](x_dec1)
        seg_outputs.append(self.final_nonlin(self.deepsupervision[1](x_dec1)))

        x_dec2 = self.upsamplers[2](x_dec1)
        x_dec2 = torch.cat((x_dec2, skips[2]), dim=1)
        x_dec2 = self.decoder[2](x_dec2)
        seg_outputs.append(self.final_nonlin(self.deepsupervision[2](x_dec2)))
        
        x_dec3 = self.upsamplers[3](x_dec2)
        x_dec3 = torch.cat((x_dec3, skips[1]), dim=1)
        x_dec3 = self.decoder[3](x_dec3)
        seg_outputs.append(self.final_nonlin(self.deepsupervision[3](x_dec3)))

        x_dec4 = self.upsamplers[4](x_dec3)
        x_dec4 = torch.cat((x_dec4, skips[0]), dim=1)
        x_dec4 = self.decoder[4](x_dec4)
        seg_outputs.append(self.final_nonlin(self.deepsupervision[4](x_dec4)))

        if self.deep_supervision and self.do_ds:
            return tuple(
                [seg_outputs[-1]] +
                [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
            )
        else:
            return seg_outputs[-1]
        
    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True, is_cls_inference=False, is_grad=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()

        self.inference_apply_nonlin = softmax_helper

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if max(mirror_axes) > 2:
                raise ValueError("mirror axes. duh")

        with autocast():
            with torch.no_grad():
                if is_cls_inference:
                    return self._internal_predict_3D_3Dconv_CLS(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                    regions_class_order, use_gaussian, pad_border_mode,
                                                                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                    verbose=verbose)
                else:
                    return self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                    regions_class_order, use_gaussian, pad_border_mode,
                                                                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                    is_grad=is_grad,
                                                                    verbose=verbose)

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool, is_grad: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data.shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            #predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)
        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            if is_grad:
                aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float64)  #lim
                aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float64) #lim
            else:
                aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
                aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map, is_grad)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    if is_grad:
                        predicted_patch[np.isinf(predicted_patch) | np.isnan(predicted_patch)] = 0
                    
                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results
    
    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps
    
    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None,
                                           is_grad: bool = False) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = torch.from_numpy(x)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float)

        if torch.cuda.is_available():
            x = x.to(f'cuda:{self.get_device()}', non_blocking=True)
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None and isinstance(mult, np.ndarray):
            mult = torch.from_numpy(mult)
            if torch.cuda.is_available():
                mult = mult.to(f'cuda:{self.get_device()}', non_blocking=True)

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1
        
        for m in range(mirror_idx):
            if is_grad:
                if m == 0:
                    '''
                    """ From nnUNet_v1 --> But we do not consider Deepsupervision in PRediction part --> We need to off DS option for prediction
                    We need to wrap this because we need to enforce self.network.do_ds = False for prediction
                    """
                    ds = self.network.do_ds
                    self.network.do_ds = False
                    '''
                    
                    pred = self.inference_apply_nonlin(self(x)) 
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, )))) 
                    result_torch += 1 / num_results * torch.flip(pred, (4,))

                if m == 2 and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )))) 
                    result_torch += 1 / num_results * torch.flip(pred, (3,))

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)))) 
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3))

                if m == 4 and (0 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )))) 
                    result_torch += 1 / num_results * torch.flip(pred, (2,))

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)))) 
                    result_torch += 1 / num_results * torch.flip(pred, (4, 2))

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)))) 
                    result_torch += 1 / num_results * torch.flip(pred, (3, 2))

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)))) 
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))
            else:
                if m == 0:
                    pred = self.inference_apply_nonlin(self(x)[1][0]) 
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, )))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (4,))

                if m == 2 and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (3,))

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3))

                if m == 4 and (0 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (2,))

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (4, 2))

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (3, 2))

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)))[1][0]) 
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch
    
    def get_device(self):
        if next(self.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index
        
    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map
    
class ResConv3dBlockRev3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple=(3, 3, 3), stride0: tuple=(1, 1, 1), stride1: tuple=(1, 1, 1), padding: tuple=(0, 0, 0), bias: bool=True) -> None:
        super().__init__()
        
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size, stride0, padding, bias=bias)
        self.instnorm0 = InstanceNorm3d(out_channels, affine=True)
        self.act0 = LeakyReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size, stride1, padding, bias=bias)
        self.instnorm1 = InstanceNorm3d(out_channels, affine=True)
        self.act1 = LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride0, bias=bias), 
            nn.InstanceNorm3d(out_channels, affine=True)
        )

    def forward(self, i):
        x = self.conv0(i)
        # if self.dropout is not None:
            # x = self.dropout(x)
        x = self.instnorm0(x)
        x = self.act0(x)

        x = self.conv1(x)
        # if self.dropout is not None:
            # x = self.dropout(x)
        x = self.instnorm1(x) + self.shortcut(i)
        x = self.act1(x)
        return x

class LinearAttention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection=0, sub_sample_flag=True, rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        self.to_qkv = DepthwiseSeparableConv3D(dim, self.inner_dim * 3)
        self.to_out = DepthwiseSeparableConv3D(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sub_sample_flag = sub_sample_flag

        # if self.rel_pos:
        #     self.relative_position_encoding = RelativePositionEmbedding3D(dim_head, reduce_size)

    def extract_voxels(self, input_tensor, n):
        B, C, D, H, W = input_tensor.size()
        
        idx_d = torch.arange(0, D, n)
        idx_h = torch.arange(0, H, n)
        idx_w = torch.arange(0, W, n)
        
        extracted_tensor = input_tensor[:, :, idx_d[:, None, None], idx_h[None, :, None], idx_w[None, None, :]]
        
        return extracted_tensor
    def forward(self, x):
        B, C, D, H, W = x.shape #([2, 32, 80, 160, 192]) # 80 40 20 10 
        reduce_size = math.ceil(min(D,H,W)/self.reduce_size)
        qkv = self.to_qkv(x) # ([2, 32, 80, 160, 192]) -> ([2, 32*3, 80, 160, 192]) dim_head * heads
        q, k, v = qkv.chunk(3, dim=1)   #([2, 32*3, 80, 160, 192]) -> ([2, 32, 80, 160, 192])  (dim_head 8, head 4)
        if self.projection == 0: #interpolate and D != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=(math.ceil(D/reduce_size), math.ceil(H/reduce_size), math.ceil(W/reduce_size)), mode='trilinear', align_corners=True), (k, v))
        elif self.projection == 1: #dilate
            k, v = map(lambda t: self.extract_voxels(t, reduce_size), (k, v))
        q = rearrange(q, 'b (dim_head heads) d h w -> b heads (d h w) dim_head', dim_head=self.dim_head, heads=self.heads, d=D, h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) d h w -> b heads (d h w) dim_head', dim_head=self.dim_head, heads=self.heads, d=math.ceil(D/reduce_size), h=math.ceil(H/reduce_size), w=math.ceil(W/reduce_size)), (k, v))#d=x.shape[-1], h=x.shape[-1], w=x.shape[-1]), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (d h w) dim_head -> b (dim_head heads) d h w', d=D, h=H, w=W, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn  
    
class NIConv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple=(3, 3, 3), stride0: tuple=(1, 1, 1), stride1: tuple=(1, 1, 1), padding: tuple=(0, 0, 0), bias: bool=True, 
                depth: int=0, num_heads: int=3, reduce_size: int=5, projection: list=None, ) -> None:
        super().__init__()
        self.depth = depth
        if self.depth > 0:
            # build blocks
            self.blocks = nn.ModuleList(
                [
                    LinearAttention3D(out_channels, heads=num_heads, dim_head=out_channels//num_heads, 
                    attn_drop=0.1, proj_drop=0.1, reduce_size=reduce_size, projection=projection[i], rel_pos=True)
                    for i in range(depth)
                ]
            )

        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size, stride0, padding, bias=bias)      
        self.act0 = LeakyReLU(inplace=True)
        self.instnorm0 = InstanceNorm3d(out_channels, affine=True)#
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size, stride1, padding, bias=bias)
        self.instnorm1 = InstanceNorm3d(out_channels, affine=True)#
        self.act1 = LeakyReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride0, bias=bias), 
            nn.InstanceNorm3d(out_channels, affine=True)
        )

    def forward(self, i):    
        x = self.conv0(i)
        # if self.dropout is not None:
            # x = self.dropout(x)
        x = self.instnorm0(x)
        x = self.act0(x)
        
        if self.depth > 0:
            for blk in self.blocks:
                out = x
                x, _ = blk(x)
                x = out+x

        x = self.conv1(x)
        # if self.dropout is not None:
            # x = self.dropout(x)
        x = self.instnorm1(x) + self.shortcut(i)
        x = self.act1(x)
        
        return x

