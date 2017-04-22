#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')

import encoder as ft_generater
import pandas as pd
import sys
import numpy as np
from numpy import array


def text_list(inputfile):

    raw_data = pd.read_csv(inputfile)
    text_data = raw_data['clean_text'].tolist()

    return text_data

def feature_generater(text_list):

    model = ft_generater.Model()
    text = text_list
    features = model.transform(text)

    return features
def label_convert(label):
    if label == 0:
        return [1, 0, 0]
    if label == 1:
        return [0, 1, 0]
    if label == 2:
        return [0, 0, 1]

def create_X_Y_data(inputfile, splitsize):

    text_data = text_list(inputfile)
    X_data = feature_generater(text_data)
    Y_data = pd.read_csv(inputfile)
    Y_data = Y_data['label'].tolist()

    X_train = array([[x] for x in X_data[0:splitsize]])
    Y_train = array([label_convert(_) for _ in Y_data[0:splitsize]])
    X_test = array([[x] for x in X_data])
    Y_test = array([label_convert(_) for _ in Y_data])


    return X_train, Y_train, X_test, Y_test

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":

    handler = logging.handlers.RotatingFileHandler(
        'find_by_terms.log', maxBytes=10 * 1024 * 1024, backupCount=10)
    logger.addHandler(handler)

    logger.info(sys.version)
    # x, y = create_X_Y_data("./twitter_pn.csv")

    X_train, X_test, Y_train, Y_test = create_X_Y_data('./twitter_pn.csv', 20)
    logger.info(X_train)
    # logger.info(Y_train)