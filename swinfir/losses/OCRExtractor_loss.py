import torch
from torch import nn as nn
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model
import sys
import cv2
import numpy as np
from torchvision import datasets, models, transforms
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from torch.autograd import Variable

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss

def load_model(path):
	with open(path + '/model.json', 'r') as f:
		json = f.read()

	model = model_from_json(json)
	model.load_weights(path + '/weights.hdf5')
        
	return model

def padding(img, min_ratio, max_ratio, color = (0, 0, 0)):
	img_h, img_w = np.shape(img)[:2]

	border_w = 0
	border_h = 0
	ar = float(img_w)/img_h

	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1
			ar = float(img_w+border_w)/(img_h+border_h)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

	border_w = border_w//2
	border_h = border_h//2

	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
	return img, border_w, border_h


class OCRFeatureExtractor(nn.Module):
    def __init__(self, OCR_path=None):
        super().__init__()
        self.to_numpy = transforms.ToPILImage()
        if OCR_path is not None:
            self.OCR_path = OCR_path
            self.OCR = load_model(self.OCR_path.as_posix())
            self.OCR = Model(self.OCR.input, self.OCR.layers[-41].output)
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(self.OCR_path.as_posix() + '/parameters.npy', allow_pickle=True).item()
            # self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            self.padding = False
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
        else:
            self.OCR_path = None

    
    def OCR_pred(self, img, fl = None, convert_to_bgr=True):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if self.padding:
            img, _, _ = padding(img, self.min_ratio, self.max_ratio, color = (127, 127, 127))        
        
        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
        img = img_to_array(img)        
        img = img.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        
        img = (img/255.0).astype('float')
        predictions = self.OCR.predict(img)
                
        return predictions

    def forward(self, images):
        batch = []
        if self.OCR_path != False:
            for img in images:
                img = np.array(self.to_numpy(img.to('cpu'))).astype('uint8')

                if self.OCR is not None:
                    features = self.OCR_pred(img)
                batch.append(features)
        logits = Variable(torch.as_tensor(batch), requires_grad=True).cuda()
        # logits = logits.view(logits.size(0), -1)
        
        return logits


@weighted_loss
def ocr_loss(SRimage, HRimage):
    path_ocr = Path('./PKU-SR')
    feature_extractor = OCRFeatureExtractor(path_ocr)
    feature_extractor.cuda()
    L1_loss =  nn.L1Loss()
    SR_features = feature_extractor(SRimage)
    HR_features = feature_extractor(HRimage)
    loss = L1_loss(SR_features, HR_features)
    return loss



@LOSS_REGISTRY.register()
class OCR_Extractor_Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(OCR_Extractor_Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, SRimage, HRimage):
        SRimage = Variable(SRimage).cuda()
        HRimage = Variable(HRimage).cuda()

        # # Move the images to GPU if available
        # if torch.cuda.is_available():
        #     imgs_LR = imgs_LR.cuda()
        #     imgs_HR = imgs_HR.cuda()

        return self.loss_weight * ocr_loss(SRimage, HRimage)