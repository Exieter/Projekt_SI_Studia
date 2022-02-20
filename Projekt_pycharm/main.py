import os
import random
import numpy as np
import cv2
#import cvlib as cv # COMMONCOUNT


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

    data = []
    for idx, entry in entry_list_csv.iterrows():
        image_path ='Train/images/' + entry['Path'] + '.png'
        image = cv2.imread(os.path.join(path, image_path))


        xml_path = 'Train/annotations/' + entry['Path'] + '.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for name in root.iter('name'):
            name.text
            if name.text == 'crosswalk':
                data.append({'image': image, 'label': 1, 'name': entry['Path'] +'.png'})
            elif name.text != 'crosswalk':
                data.append({'image': image, 'label': 0, 'name': entry['Path'] +'.png'})


    return data





def load_data_train(path, filename):
    """
    Loads data from disk.
    @param path: Path to dataset directory.
    @param filename: Filename of csv file with information about samples.
    @return: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    """
    entry_list_csv = pandas.read_csv(os.path.join(path, filename))

    data = []
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
                data.append({'image': image, 'label': 1})

    return data


def learn_bovw(data):
    """
    Learns BoVW dictionary and saves it as "voc.npy" file.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Nothing
    """
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)


def extract_features(data):
    """
    Extracts features for given data and saves it as "desc" entry.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Data with added descriptors for each sample.
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        # compute descriptor and add it as "desc" entry in sample
        # TODO PUT YOUR CODE HERE
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        sample['desc'] = desc
        # ------------------

    return data



def train(data):
    """
    Trains Random Forest classifier. #jak to zaimplementowac to random forest
    #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Trained model.
    """
    # train random forest model and return it from function.
    # TODO PUT YOUR CODE HERE

    descs = []
    labels = []

    for sample in data:
        if sample['desc'] is not None:
            descs.append(sample['desc'].squeeze(0))
            labels.append(sample['label'])

    rf = RandomForestClassifier()
    rf.fit(descs, labels)
    # ------------------

    return rf


def predict(rf, data, data2):
    """
    Predicts labels given a model and saves them as "label_pred" (int) entry for each sample.
    @param rf: Trained model.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Data with added predicted labels for each sample.
    """
    # perform prediction using trained model and add results as "label_pred" (int) entry in sample
    # TODO PUT YOUR CODE HERE

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = int(pred)


            print(sample['name'])

            #box, label, count = cv.detect_common_objects(sample['image'], model = rf)
            #print(count)
            #print('box:', box[0])

    return data





def balance_dataset(data, ratio):
    """
    Subsamples dataset according to ratio.
    @param data: List of samples.
    @param ratio: Ratio of samples to be returned.
    @return: Subsampled dataset.
    """
    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data

def main():
    data_train = load_data_train('./', 'Train.csv')
    data_train = balance_dataset(data_train, 1.0)

    data_test = load_data_test('./', 'Train.csv')
    data_test = balance_dataset(data_test, 1.0)


    #print('learning BoVW')
    #learn_bovw(data_train)

    #print('extracting train features')
    data_train = extract_features(data_train)


    #print('training')
    rf = train(data_train)

    #print('extracting test features')
    data_test2 = extract_features(data_test)

    #print('testing on testing dataset')
    data_test = predict(rf, data_test2, data_test)
    return



if __name__ == '__main__':
    main()