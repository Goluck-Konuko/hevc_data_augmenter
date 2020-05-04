from tqdm import tqdm
from argparse import ArgumentParser
from augmenter import offline_augmenter
import numpy as np
import cv2
import glob
import os

data_list = glob.glob("test_data/src_odps/*.png")
mode_data = True



if __name__ == "__main__":
    parser= ArgumentParser()
    parser.add_argument("--d",dest="dataset_path",required=True, help="Path to dataset")
    parser.add_argument("--m",dest="mode_data", default=False, help="True || False -modes stores in the first pixel of the patches")
    parser.add_argument("--o", dest="output_path", default="/augmentation_results/",help="path to store the results")
    args = parser.parse_args()

    data_list = glob.glob(args.dataset_path +"*.png")
    mode_data = args.mode_data

    try:
        os.mkdir(args.output_path)
    except FileExistsError:
        try:
            os.mkdir(args.output_path+"/offline/")
        except FileExistsError:
            if mode_data:
                try:
                    os.mkdir(args.output_path+"/offline/mode_data/")
                except FileExistsError:
                    print("All folders ready")
                    print("Computing augmentation data")
            else:
                try:
                    os.mkdir(args.output_path+"/offline/mode_data_off/")
                except FileExistsError:
                    print("All folders ready")
                    print("Computing augmentation data")

    except Exception as err:
        print("Directory err: ",err)
    
    if mode_data:
        offline_augmenter(data_list, output_path = args.output_path + "offline/mode_data/")
    else:
        offline_augmenter(data_list, output_path = args.output_path + "offline/mode_data_off/")


