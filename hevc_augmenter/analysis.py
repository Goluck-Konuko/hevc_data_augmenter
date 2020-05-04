from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import glob
import os
import cv2



def compute_metrics(original_pu, hevc_pu):
    mse = np.sum((original_pu - hevc_pu) ** 2)/ 32**2
    psnr = 10*np.log10(255**2/mse)
    return mse, psnr

if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--d", dest="dataset_path",required=True, help="Path to folder with ODP patches")

    args = parser.parse_args()

    dataset = glob.glob(args.dataset_path + "/*.png")

    metrics_mse = []
    metrics_psnr = []

    for odp in tqdm(dataset):
        patch = cv2.imread(odp, cv2.IMREAD_GRAYSCALE)
        original_pu = patch[32:, 32:64]
        hevc_pu = patch[32:,160:]
        mse, psnr = compute_metrics(original_pu, hevc_pu)

        metrics_mse.append(mse)
        metrics_psnr.append(psnr)


    mean_mse = np.mean(np.array(metrics_mse))
    
    mean_psnr = np.mean(np.array(metrics_psnr))
    

    std_mse = np.std(np.array(metrics_mse))
    std_psnr = np.std(np.array(metrics_psnr))
    print("Original Patches")
    print("MEAN MSE: ",mean_mse)
    print("Standard Deviation MSE: ",std_mse)
    print("---------------------------")
    print("MEAN PSNR: ", mean_psnr)
    print("Standard Deviation PSNR: ",std_psnr)

    plt.scatter(range(len(dataset)),metrics_mse, label="MSE")
    plt.scatter(range(len(dataset)), metrics_psnr, label="PSNR")
    plt.legend()
    plt.xlabel("Patches")
    plt.ylabel("PSNR/MSE")
    plt.title("Augmentation Patches: Offline- no mode Information")
    plt.savefig("analytics/aug_patches_offline_metrics.png")
    plt.close()







