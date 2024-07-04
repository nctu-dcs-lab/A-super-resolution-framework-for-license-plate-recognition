import re
import cv2
import torch
import functions
import numpy as np
import torchvision.transforms as transforms
import os
from pathlib import Path
from torch import nn
from keras.preprocessing.image import img_to_array
import csv
from PIL import Image
from tqdm import tqdm

class OCRloss(nn.Module):
    def __init__(self, OCR_path=None):
        super().__init__()
        self.to_numpy = transforms.ToPILImage()
        if OCR_path is not None:
            self.OCR_path = OCR_path
            self.OCR = functions.load_model(OCR_path.as_posix())
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(OCR_path.as_posix() + '/parameters.npy', allow_pickle=True).item()
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            self.padding = True
            self.aspect_ratio = self.IMAGE_DIMS[1] / self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
        else:
            self.OCR_path = None

    def OCR_pred(self, img, convert_to_bgr=True):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if self.padding:
            img, _, _ = functions.padding(img, self.min_ratio, self.max_ratio, color=(127, 127, 127))
        
        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
        img = img_to_array(img)        
        img = img.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        
        img = (img / 255.0).astype('float')
        predictions = self.OCR.predict(img)

        plates = [''] * 1
        for task_idx, pp in enumerate(predictions):
            idx_aux = task_idx + 1
            task = self.tasks[task_idx]

            if re.match(r'^char[1-9]$', task):
                for aux, p in enumerate(pp):
                    plates[aux] += self.ocr_classes['char{}'.format(idx_aux)][np.argmax(p)]
            else:
                raise Exception('unknown task \'{}\'!'.format(task))
                
        return plates
    
    def levenshtein(self, a, b):
        if not a: 
            return len(b)

        if not b: 
            return len(a)

        return min(self.levenshtein(a[1:], b[1:]) + (a[0] != b[0]), self.levenshtein(a[1:], b) + 1, self.levenshtein(a, b[1:]) + 1)

    def forward(self, SR, HR):
        dists = 0
        count = 0

        if self.OCR_path is not None:
            for imgSR, imgHR in zip(SR, HR):
                imgSR = np.array(self.to_numpy(imgSR.to('cpu'))).astype('uint8')
                imgHR = np.array(self.to_numpy(imgHR.to('cpu'))).astype('uint8')
                
                if self.OCR is not None:
                    predSR = self.OCR_pred(imgSR)[0]
                    predHR = self.OCR_pred(imgHR)[0]
                    dists += torch.as_tensor(self.levenshtein(predHR, predSR) / 7)
                count += 1
            dists /= count
        
        return dists

def get_ground_truth_SR(hr_folder, sr_filename):
    # Replace "_SR.png" with ".txt" to get the corresponding ground truth file name

    txt_filename = sr_filename.replace("_SR.png", ".txt")

    # only for 0503_test in lpr-rsr
    # txt_filename = sr_filename.lstrip("SR_").replace(".png", ".txt")
    # txt_filename = sr_filename.lstrip("SR_").replace(".jpg", ".txt")

    # Construct the full path to the ground truth file in the HR folder
    txt_path = os.path.join(hr_folder, txt_filename)
    # Read the content of the ground truth file
    with open(txt_path, "r") as file:
        ground_truth = file.read().strip()
    return ground_truth

def get_ground_truth_HR(hr_folder, sr_filename):
    # Replace "_SR.png" with ".txt" to get the corresponding ground truth file name
    # txt_filename = sr_filename.replace(".png", ".txt")
    txt_filename = sr_filename.replace(".jpg", ".txt")
    # Construct the full path to the ground truth file in the HR folder
    txt_path = os.path.join(hr_folder, txt_filename)
    # Read the content of the ground truth file
    with open(txt_path, "r") as file:
        ground_truth = file.read().strip()
    return ground_truth

def process_images(sr_folder, hr_folder, ocr_loss, output_csv):
    results = []
    total_accuracy_sr = 0

    # List all files in the SR folder with a ".png" extension
    sr_files = [file for file in os.listdir(sr_folder) if file.endswith('.png')]
    # sr_files = [file for file in os.listdir(sr_folder) if file.endswith('.jpg')]

    # only for 0503_test and lpr-on-PKU-test
    # sr_files = [file for file in os.listdir(sr_folder) if file.startswith('SR_') and file.endswith('.png')]
    # sr_files = [file for file in os.listdir(sr_folder) if file.startswith('SR_') and file.endswith('.jpg')]

    for sr_filename in tqdm(sr_files):
        # Construct the full path to the SR image
        sr_path = os.path.join(sr_folder, sr_filename)
        # Read the SR image
        sr_img = cv2.imread(sr_path)

        # Get the ground truth from the corresponding HR folder
        # ground_truth = get_ground_truth_HR(hr_folder, sr_filename)
        ground_truth = get_ground_truth_SR(hr_folder, sr_filename)


        # Perform OCR prediction on the SR image
        sr_pred = ocr_loss.OCR_pred(sr_img, convert_to_bgr=True)[0]
        # Calculate Levenshtein distance and accuracy for SR prediction
        distance_sr = ocr_loss.levenshtein(ground_truth, sr_pred)
        accuracy_sr = 7 - (distance_sr - 1)
        total_accuracy_sr += accuracy_sr
        # Append the results to the list
        results.append([ground_truth, os.path.basename(sr_path), sr_pred, accuracy_sr])
    # Calculate average accuracy for SR images
    avg_accuracy_sr = total_accuracy_sr / len(sr_files)
    # Append average accuracy for SR images to the results
    results.append(['Average', '', '', avg_accuracy_sr])
    # Write the results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['GT Plate', 'file', 'SR Prediction', 'Accuracy (SR)'])
        csv_writer.writerows(results)

if __name__ == "__main__":
    
    # Get the current directory
    current_directory = os.getcwd()

    # Define the relative path to the sr_folder
    # relative_path = "SwinFIR/results/SwinFIR_SRx2"

    # Construct the absolute path by joining the current directory with the relative path
    # sr_folder = os.path.abspath('./results/trainOnPKU_char_x4/visualization/LP')
    # sr_folder = os.path.abspath('./results/trainOnPKU_char_x4/visualization/PKU_dataset')

    # sr_folder = os.path.abspath('../lpr-rsr-ext/Proposed/0503_test') 
    # sr_folder = os.path.abspath('../lpr-rsr-ext/Proposed/lpr-on-PKU-test')

    # sr_folder = os.path.abspath('./results/LR')
    # sr_folder = "../clear_PSrec/HR/val2"

    #sr_folder = "../lpr-rsr-ext/clear_PSrec/LR"

    sr_folder = os.path.abspath('./results/PKU_VGG_CRNN_cos/visualization/PKU_dataset_test')
    hr_folder = "../lpr-rsr-ext/SRPlates/PKU_dataset/HR_new"  # Replace with the path to your HR text file folder

    # sr_folder = os.path.abspath('../lpr-rsr-ext/SRPlates/PKU_dataset/LR_new/test')


    # hr_folder = "../lpr-rsr-ext/clear_PSrec/HR"  # Replace with the path to your HR text file folder
   

    OCR_path = Path('./PKU-SR')  # Replace with the path to your OCR model
    output_csv = "PKU_VGG_CRNN_cos.csv"  # Output CSV file

    # Initialize OCR loss
    ocr_loss = OCRloss(OCR_path=OCR_path)

    # Process SR images
    process_images(sr_folder, hr_folder, ocr_loss, output_csv)
