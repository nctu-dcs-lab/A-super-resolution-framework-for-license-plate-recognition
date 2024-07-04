import torch.nn as nn
import torch
import torch.nn.functional as F

class myNet_ocr(nn.Module):
    def __init__(self,cfg=None,num_classes=78,export=False, extract_layers=None):
        super(myNet_ocr, self).__init__()
        if cfg is None:
            cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256]

        self.feature = self.make_layers(cfg, True)
        self.export = export
        self.loc =  nn.MaxPool2d((5, 2), (1, 1),(0,1),ceil_mode=False)
        self.newCnn=nn.Conv2d(cfg[-1],num_classes,1,1)

        # Define the layers from which to extract features
        if extract_layers is None:
            extract_layers = ['conv1', 'conv2', 'conv3']
        self.extract_layers = extract_layers
       
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d =nn.Conv2d(in_channels, cfg[i], kernel_size=5,stride =1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else :
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1,1),stride =1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):

        features = {}
        for name, layer in self.feature.named_children():
            x = layer(x)
            if name in self.extract_layers:
                features[name] = x.clone()
            # print(f'Intermediate Tensor Size after {name}: {x.size()}')

        # x = self.feature(x)

        # x=self.loc(x)
        # x=self.newCnn(x)
        
        # if self.export:
        #     conv = x.squeeze(2) # b *512 * width
        #     conv = conv.transpose(2,1)  # [w, b, c]
        #     return conv, features
        # else:
        #     b, c, h, w = x.size()
        #     assert h == 1, "the height of conv must be 1"
        #     conv = x.squeeze(2) # b *512 * width
        #     conv = conv.permute(2, 0, 1)  # [w, b, c]
        #     output = F.log_softmax(conv, dim=2)
        
            # return output, features
        return features

# if __name__ == '__main__':
#     x = torch.randn(1,3,48,168)
#     cfg =[32,'M',64,'M',128,'M',256]
#     extract_layers = ['conv1', 'conv2', 'conv3']
#     model = myNet_ocr(num_classes=78, export=True, cfg=cfg, extract_layers=extract_layers)
#     out, features = model(x)
    # print(out.shape)

    # Accessing intermediate features
    # for layer_name, feature_map in features.items():
    #     print(f'Layer: {layer_name}, Shape: {feature_map.shape}')