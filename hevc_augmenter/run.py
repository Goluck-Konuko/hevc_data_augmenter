from argparse import ArgumentParser
from augmenter import online_augmenter, offline_augmenter
import glob
import cv2
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode",dest ="aug_mode",default="online",help="Data augmentation mode: online || offline")
    parser.add_argument("--dataset", dest="dataset_path", default=None, help="(Required when mode is offline)Path to dataset for offline augmentation")
    parser.add_argument("--patch", dest="input_path", default=None, help="Path to odp patch")
    parser.add_argument("--output",dest="output_path", default="test_data/pred_results/test1/",help="Path to output folder")

    args = parser.parse_args()

    if args.dataset_path ==None and args.input_path== None:
        print("Specify path to dataset of patch to be augmented")
    else:
        if args.aug_mode== "online":

            #perform online augmentation- read one patch and return its augmentation
            if args.input_path == None:
                print("Missing argument: Include \"--patch path/to/odp_patch\" for online augmentation")
            else:
                patch = args.input_path.split('/')
                patch_name = patch[len(patch)-1]
                odp = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
                output_patch = online_augmenter(odp_patch = odp, diskpath=args.output_path)
                
                cv2.imwrite(args.output_path + 'aug_online_' +patch_name, output_patch)
        else:
            #perform offline augmentation
            if args.dataset_path ==None:
                print("Missing argument: A path to dataset folder must be specified for this operation")
            else:
                #parse image directory
                dataset = glob.glob(args.dataset_path + "/*.png")
                #create output directory
                try:
                    offline_augmenter(odp_batch = dataset, multiplier = 5, output_path = args.output_path)
                except Exception as err:
                    print("Error: ", err)