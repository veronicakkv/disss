
import os
import zipfile
from os import listdir
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
# with zipfile.ZipFile("breast_cancer_images.zip", 'r') as zip_ref:
#     zip_ref.extractall("breast_cancer_images")


def extract_all_image_paths(input_path=Path.cwd()/'breast_cancer_images'):
    print(input_path)
    image_paths = [image_path for image_path in Path.glob(
        input_path, pattern='*/*/*.png')]
    return image_paths


def get_base_path():
    base_path = 'breast_cancer_images/IDC_regular_ps50_idx5'
    folder = os.listdir(base_path)
    len(folder)
    return base_path, folder


def get_total_images(base_path, folder):
    total_images = 0
    for x in range(len(folder)):
        patient_id = folder[x]
        for num in [0, 1]:
            patient_path = base_path + "/" + patient_id
            class_path = patient_path + "/" + str(num) + "/"
            subfiles = os.listdir(class_path)
            total_images += len(subfiles)
    print(total_images)
    return total_images


def get_dataframe(base_path, folder, total_images):
    data = pd.DataFrame(index=np.arange(0, total_images),
                        columns=["patient_id", "path", "target"])
    k = 0
    for n in range(len(folder)):
        patient_id = folder[n]
        patient_path = base_path + "/" + patient_id
        for c in [0, 1]:
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = os.listdir(class_path)
            for m in range(len(subfiles)):
                image_path = subfiles[m]
                data.iloc[k]["path"] = class_path + image_path
                data.iloc[k]["target"] = c
                data.iloc[k]["patient_id"] = patient_id
                k += 1

    data.head()
    return data

# image_paths = extract_all_image_paths()
# base_path, folder = get_base_path()
# get_total_images(base_path, folder)
