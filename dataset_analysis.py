
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
# with zipfile.ZipFile("breast_cancer_images.zip", 'r') as zip_ref:
#     zip_ref.extractall("breast_cancer_images")

# This function gets the paths of all the images


def extract_all_image_paths(input_path=Path.cwd()/'breast_cancer_images'):
    print(input_path)
    image_paths = [image_path for image_path in Path.glob(
        input_path, pattern='*/*/*.png')]
    return image_paths

# This function gets the base paths of all the images


def get_base_path():
    base_path = 'breast_cancer_images/IDC_regular_ps50_idx5'
    folder = os.listdir(base_path)
    len(folder)
    return base_path, folder

# This function gets the number of all the images


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


# This function gputs the path, target and patient_id into a dataframe
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


"""
This function select n samples of the dataframe  
"""


def get_cancer_dfs(data, plot=False, num=25000):
    # Put all the cancer class in a separate dataset.
    cancer_df = data.loc[data['target'] == 1].sample(n=num, random_state=42)

    # Randomly select 25000 observations from the non cancerous
    non_cancer_df = data.loc[data['target']
                             == 0].sample(n=num, random_state=42)

    # Concatenate both dataframes again
    normalized_df = pd.concat([cancer_df, non_cancer_df])
    if plot == True:
        # plot the dataset after the undersampling
        plt.figure(figsize=(8, 8))
        sns.countplot('target', data=normalized_df)
        plt.title('Balanced Classes')
        plt.show()
    return normalized_df


def get_train_test_patient_ids(normalized_df):
    patients = normalized_df.patient_id.unique()
    print(len(patients))

    train_ids, test_ids = train_test_split(patients,
                                           test_size=0.3,
                                           random_state=3)
    # print(len(train_ids), len(test_ids))
    return train_ids, test_ids


def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:, "x"].str.replace(
        "x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:, "y"].str.replace(
        "y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df


def get_train_test_dfs(normalized_df, train_ids, test_ids, plot=False):
    train_df = normalized_df.loc[normalized_df.patient_id.isin(
        train_ids), :].copy()

    test_df = normalized_df.loc[normalized_df.patient_id.isin(
        test_ids), :].copy()

    train_df = extract_coords(train_df).sample(frac=1)
    test_df = extract_coords(test_df).sample(frac=1)
    plt.figure(figsize=(8, 8))
    if plot == True:
        sns.countplot('target', data=test_df)
        plt.title('Balanced Classes')
        plt.show()
        return train_df, test_df

# image_paths = extract_all_image_paths()
# base_path, folder = get_base_path()
# get_total_images(base_path, folder)
