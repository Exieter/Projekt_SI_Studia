import os
import random
import numpy as np
import cv2

from sklearn.ensemble import RandomForestClassifier
import pandas
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix


import xml.etree.ElementTree as ET #biblioteka do czytania .xml

def load_data_test(path, filename):
    """
    Loads data from disk.
    @param path: Path to dataset directory.
    @param filename: Filename of csv file with information about samples.
    @return: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    """
    entry_list_csv = pandas.read_csv(os.path.join(path, filename))

    data_test = []
    for idx, entry in entry_list_csv.iterrows():
        image_path ='Train/images/' + entry['Path'] + '.png'
        image = cv2.imread(os.path.join(path, image_path))


        xml_path = 'Train/annotations/' + entry['Path'] + '.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for name in root.iter('name'):
            name.text
            if name.text == 'crosswalk':
                data_test.append({'image': image, 'label': name.text})
            elif name.text != 'crosswalk':
                data_test.append({'image': image, 'label': other})
    return data_test





def load_data_train(path, filename):
    """
    Loads data from disk.
    @param path: Path to dataset directory.
    @param filename: Filename of csv file with information about samples.
    @return: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    """
    entry_list_csv = pandas.read_csv(os.path.join(path, filename))

    data_train = []
    for idx, entry in entry_list_csv.iterrows():
        image_path = 'Train/images/' + entry['Path'] + '.png'
        image = cv2.imread(os.path.join(path, image_path))


# Jesli zdjecie jest crosswalk wtedy data.append
        xml_path = 'Train/annotations/' + entry['Path'] + '.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for name in root.iter('name'):
            name.text
            if name.text == 'crosswalk':
                data_train.append({'image': image, 'label': name.text})

    return data_train



def main():
    #data_train = load_data_test('./', 'Train.csv')
    load_data_train('./', 'Train.csv')
    return


if __name__ == '__main__':
    main()