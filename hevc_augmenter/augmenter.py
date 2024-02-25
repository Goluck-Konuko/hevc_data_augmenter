from .hevc_predictor import Predictor
import numpy as np
from tqdm import tqdm
import random
import cv2

def offline_augmenter(odp_batch=None,multiplier=5, output_path = None, mode_data=False):
    """
        Computes structural similarity and mse metrics to return X best augmentation patches.
        specify X as multiplier.
        If multiplier==1, the offline augmenter behaves like the online version but much slower.
    """
    if odp_batch ==None:
        print("Error: missing list of odp patch names")
    else:
        if mode_data:
            for patch in tqdm(odp_batch):
                augment = random.choice([True, False])
                if augment:
                    odp_patch = cv2.imread(patch, cv2.IMREAD_GRAYSCALE)
                    mode = odp_patch[0,0]
                    data_generator = Predictor(odp = odp_patch,multiplier=multiplier, diskpath= output_path)
                    if mode == 2: #DC prediction - augment with planar prediction
                        pred = data_generator.predict_one(mode = 1)
                    if mode == 1: # Planar prediction - augment with DC prediction
                        pred = data_generator.predict_one(mode = 0)
                    else:#other prediction directions are augmented with their neighbors
                        if mode == 3:
                            pred = data_generator.predict_one(mode = 3)
                        if mode == 35:
                            pred = data_generator.predict_one(mode = 34)
                        else:
                            pred = data_generator.predict_one(mode = np.random.choice([mode+1, mode-1])-1) 
                    out = output_path+ "aug_offline_"+ patch.split('\\')[len(patch.split('\\'))-1]
                    cv2.imwrite(out, pred)
        else:
            for patch in tqdm(odp_batch):
                augment = random.choice([True, False])
                if augment:
                    odp_patch = cv2.imread(patch, cv2.IMREAD_GRAYSCALE)
                    data_generator = Predictor(odp = odp_patch)
                    pred = data_generator.predict_all()
                    out = output_path+ "aug_offline_"+ patch.split('\\')[len(patch.split('\\'))-1]
                    cv2.imwrite(out, pred)

def online_augmenter(odp_patch=None, diskpath=None, mode_data=False):
    '''
        Returns a patch with closest structural similarity to the current prediction mode
        i.e. one of the results of neighboring prediction modes
    '''
    if mode_data:
        mode = odp_patch[0,0]
        data_generator = Predictor(odp = odp_patch, diskpath= diskpath)
        if mode == 2: #DC prediction - augment with planar prediction
            pred = data_generator.predict_one(mode = 1)
        if mode == 1: # Planar prediction - augment with DC prediction
            pred = data_generator.predict_one(mode = 0)
        else:#other prediction directions are augmented with their neighbors
            if mode == 3:
                pred = data_generator.predict_one(mode = 3)
            if mode == 35:
                pred = data_generator.predict_one(mode = 34)
            else:
                pred = data_generator.predict_one(mode = np.random.choice([mode+1, mode-1])-1) 
    else:
        data_generator = Predictor(odp = odp_patch)
        pred = data_generator.predict_all(select=True)
    return pred
