from tqdm import tqdm
from argparse import ArgumentParser
from augmenter import online_augmenter
import numpy as np
import cv2
import glob
import os


if __name__ == "__main__":
    parser= ArgumentParser()
    parser.add_argument("--d",dest="dataset_path",required=True, help="Path to dataset")
    parser.add_argument("--m",dest="mode_data", default=False, help="True || False -modes stores in the first pixel of the patches")
    parser.add_argument("--o", dest="output_path", default="/augmentation_results/",help="path to store the results")
    args = parser.parse_args()

    data_list = glob.glob(args.dataset_path + "*.png")
    mode_data = args.mode_data


    try:
        os.mkdir(args.output_path)
    except FileExistsError:
        try:
            os.mkdir(args.output_path+"online/")
        except FileExistsError:
            if mode_data:
                try:
                    os.mkdir(args.output_path+"online/mode_data/")
                except FileExistsError:
                    print("All folders ready")
                    print("Computing augmentation data")
            else:
                try:
                    os.mkdir(args.output_path+"online/mode_data_off/")
                except FileExistsError:
                    print("All folders ready")
                    print("Computing augmentation data")

    except Exception as err:
        print("Directory err: ",err)


    for patch_name in tqdm(data_list):
        patch = patch_name.split('\\')[len(patch_name.split('\\'))-1]
        odp_patch = np.array(cv2.imread(patch_name, cv2.IMREAD_GRAYSCALE))
    
        if mode_data:
            output_patch = online_augmenter(odp = odp_patch, mode_data= mode_data)
            cv2.imwrite(args.output_path+ "online/mode_data/" + 'aug_online_' +patch, output_patch)
        else:
            output_patch = online_augmenter(odp = odp_patch)
            cv2.imwrite(args.output_path+"online/mode_data_off/" + 'aug_online_' +patch, output_patch)

    
    