#!/bin/bash
import os
import sys

import numpy as np
from argparse import ArgumentParser
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nnmodel.export import load_data, load_model
import joblib

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

weights_path = {
    "lr": "./lr.pkl",
    "svc": "./svc.pkl",
    "bg": "./bg.pkl",
}

def parse_arg(argv):
    parser = ArgumentParser(description='Main Program...')
    parser.add_argument('-s', '--save_checkpoint', type=bool, default=True, help='Save checkpoint or not')
    return parser.parse_args(argv)


def _load_and_flatten(get_rate=1.0):
    # if first time, set is_download=True
    train_data, test_data = load_data(is_download=False)

    train_data = train_data.data.cpu().numpy(), train_data.targets.cpu().numpy()
    test_data = test_data.data.cpu().numpy(), test_data.targets.cpu().numpy()



    # flatten
    img_shape = train_data[0].shape[1:]
    img_size = img_shape[0] * img_shape[1]
    n = train_data[0].shape[0]
    m = test_data[0].shape[0]
    train_data[0].resize((n, img_size))
    test_data[0].resize((m, img_size))
    return train_data, test_data


def evaluate(y_true, y_pred):
    """
    return acc, precision, recall, f1
    :param y_pred:
    :param y_true:
    :return:
    """
    return accuracy_score(y_true, y_pred), \
        precision_score(y_true, y_pred, average='micro'), \
        recall_score(y_true, y_pred, average='micro'), \
        f1_score(y_true, y_pred, average='micro')


def plot_curve(evals):

    pass


def main(argv):
    arg = parse_arg(argv)

    # Load data
    train_data, test_data = _load_and_flatten()
    X_train, y_train = train_data
    X_test, y_test = test_data


    # Initialize the classifiers
    classifier1 = LogisticRegression(max_iter=1000)
    classifier2 = SVC(max_iter=1000)
    # Classifier 3 does not need to train, if it has pretrained
    classifier3 = load_model(device)
    classifier4 = BaggingClassifier(estimator=SVC(), n_estimators=3)

    # load weights
    if os.path.exists(weights_path['lr']) and os.path.exists(weights_path['svc'])\
            and os.path.exists(weights_path['bg']):
        classifier1 = joblib.load(weights_path['lr'])
        classifier2 = joblib.load(weights_path['svc'])
        classifier4 = joblib.load(weights_path['bg'])
        print("Model load successfully")
    else:
        # Train the classifiers
        classifier1.fit(X_train, y_train)
        classifier2.fit(X_train, y_train)
        classifier4.fit(X_train, y_train)

        # Save checkpoint
        if arg.save_checkpoint:
            joblib.dump(classifier1, "lr.pkl")
            joblib.dump(classifier2, "svc.pkl")
            joblib.dump(classifier4, "bg.pkl")
            print("Save Weights Successfully")
            pass

    # Make predictions on the testing set
    y_pred3 = classifier3.predict(X_test)

    y_pred1 = classifier1.predict(X_test)
    y_pred2 = classifier2.predict(X_test)
    y_pred4 = classifier4.predict(X_test)

    # Calculate the accuracy of each individual classifier
    indicator_cnn = evaluate(y_test, y_pred3)
    indicator_lr = evaluate(y_test, y_pred1)
    indicator_svc = evaluate(y_test, y_pred2)
    indicator_bagging = evaluate(y_test, y_pred4)

    print(indicator_cnn)
    print(indicator_lr)
    print(indicator_svc)
    print(indicator_bagging)

if __name__ == '__main__':
    argv = sys.argv
    main(argv[1:])