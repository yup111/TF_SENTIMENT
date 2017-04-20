#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')

import pandas as pd
import sys
from numpy import array



def label_convert(label):
    if label == 0:
        return 0
    if label == 1:
        return 1
    if label == 2:
        return 1

def create_X_Y_data(inputfile, splitsize):

    raw_data = pd.read_csv(inputfile)
    X_data = raw_data['clean_text'].tolist()
    Y_data = raw_data['label'].tolist()
    Y_data = [label_convert(_) for _ in Y_data]


    return X_data, Y_data


if __name__ == "__main__":

    handler = logging.handlers.RotatingFileHandler(
        'find_by_terms.log', maxBytes=10 * 1024 * 1024, backupCount=10)
    logger.addHandler(handler)

    logger.info(sys.version)

    X_train, Y_test = create_X_Y_data('./twitter_pn.csv', 20)
    logger.info(Y_test)
    logger.info(Y_train)