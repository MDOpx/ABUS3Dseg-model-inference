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
import torch


def load_pretrained_weights(network, fname, verbose=False, load_only_encoders:bool=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if ('coder' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        if load_only_encoders:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'encoder.' in k}
            print("Loading only Encoder weights...")
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            print("Loading Full weights...")
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")

        network.load_state_dict(model_dict)
    else:
        try:
            converted_dict = {}
            for k, v in pretrained_dict.items():
                if 'encoder.encoders.5.' in k:
                    k = k.replace('encoder.encoders.5.', 'bottleneck.')
                if 'encoder.encoders.' in k:
                    k = k.replace('encoders.', '')
                if 'classification_head' in k:
                    k = k.replace('classification_head.', 'cls_head.')
                if 'seg_decoder.decoders' in k:
                    k = k.replace('seg_decoder.decoders.', 'seg_decoder.decoder.')
                if 'decoders.' == k[:9]:
                    k = k.replace('decoders.', 'decoder.')
                if 'encoders.' == k[:9]:
                    k = k.replace('encoders.', 'encoder.')
                if 'seg_decoder.num_classes' in k:
                    continue
                converted_dict[k] = v

            if load_only_encoders:
                converted_dict = {k: v for k, v in converted_dict.items() if 'encoder.' in k}
                print("Loading only Encoder weights...")
                model_dict.update(converted_dict)
            else:
                print("Loading Full weights...")
                model_dict.update(converted_dict)
            
            print("################### Loading pretrained weights from file ", fname, '###################')
            network.load_state_dict(model_dict)
        except:
            raise RuntimeError("Pretrained weights are not compatible with the current network architecture")

