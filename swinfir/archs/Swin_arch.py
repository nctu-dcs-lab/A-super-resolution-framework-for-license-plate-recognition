import torch
from torch import nn
from torchvision.transforms import InterpolationMode
# from torchvision.models.swin_transformer import SwinTransformer
# from torchvision.models.utils import register_model, WeightsEnum, Weights
from typing import List, Optional, Callable, Any
from basicsr.utils.registry import ARCH_REGISTRY
import os

SWIN_T_PRETRAIN_PATH = 'experiments/pretrained_models/swin_t-704ceda3.pth' 

import torchvision.models as models
model = models.swin_t(pretrained = True)

NAMES = {
    'swin_t': [
        'layers.0.blocks.0', 'layers.0.blocks.1', 
        'layers.1.blocks.0', 'layers.1.blocks.1', 
        'layers.2.blocks.0', 'layers.2.blocks.1', 'layers.2.blocks.2', 'layers.2.blocks.3', 'layers.2.blocks.4', 'layers.2.blocks.5', 
        'layers.3.blocks.0', 'layers.3.blocks.1'
    ]
}

@ARCH_REGISTRY.register()
class SwinTFeatureExtractor(nn.Module):
    """SwinT network for feature extraction.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of SwinT network will be
            optimized. Default: False.
        
    """
    def __init__(self,
                 layer_name_list,
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 progress: bool = True,
                 **kwargs: Any):
        super(SwinTFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.model = model

        # self.model = SwinTransformer(
        # patch_size=[4, 4],
        # embed_dim=96,
        # depths=[2, 2, 6, 2],
        # num_heads=[3, 6, 12, 24],
        # window_size=[7, 7],
        # stochastic_depth_prob=0.2,
        # **kwargs,
        # )

        self.names = NAMES['swin_t']

        # only borrow layers that will be used to avoid unused params
        # max_idx = 0
        # for v in layer_name_list:
        #     idx = self.names.index(v)
        #     if idx > max_idx:
        #         max_idx = idx
    
        max_idx = max([self.names.index(layer) for layer in layer_name_list])
        self.features_module = nn.Sequential(*list(self.model.features.children())[:max_idx + 1])

        #model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

        if not requires_grad:
            self.model.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # features = {}
        # for name, module in self.model.features.named_children():
        #     x = module(x)
        #     if name in self.layers:
        #         features[name] = x.clone()
        # return features

        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}

        # print("Layer name list:", self.layer_name_list)
        # print("Model keys:", list(self.model._modules.keys()))
        # print("self.features_module:", self.features_module)
        # print("self.features_module.named_children(): ", self.features_module.named_children())
        # print("self.model.features._modules.items() ", self.model.features._modules.items())



        # features_module = self.model.features
        #print("features_module: ",features_module)

        #for key, layer in features_module._modules.items():
        # for key, layer in self.features_module.named_children():
        #     x = layer(x)
        #     if key in self.layer_name_list:
        #         output[key] = x.clone()

        # Iterate through the layers in the features_module
        for idx, layer in enumerate(self.features_module):
            x = layer(x)
            layer_name = self.names[idx]
            if layer_name in self.layer_name_list:
                output[layer_name] = x.clone()

        
        # print("Output dictionary:", output)
        # print("Type of output:", type(output))
        # print("Type of output[key]:", type(output['layers.2.blocks.5']))
        # print("Output[key] shape:", output['layers.2.blocks.5'].shape)


        return output

