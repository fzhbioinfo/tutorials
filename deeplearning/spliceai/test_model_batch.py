import h5py
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from train_model import spliceai
from tensorflow.keras.models import load_model


def build_model(weights):
    filters = [32] * 4 + [32] * 4 + [32] * 4 + [32] * 4
    kernel_size = [11] * 4 + [11] * 4 + [21] * 4 + [41] * 4
    dilation_rate = [1] * 4 + [4] * 4 + [10] * 4 + [25] * 4
    model = spliceai(filters, kernel_size, dilation_rate)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(weights)
    return model


def run(args):
    h5f = h5py.File(args.dataset, 'r')
    total = len(h5f.keys()) // 2
    if args.weights:
        model = build_model(args.weights)
    else:
        model = load_model(args.model)
    accuracy = CategoricalAccuracy()
    entropy = CategoricalCrossentropy()
    X, Y = [], []
    for i in range(total):
        x = h5f[f'X{i}'][:]
        y = h5f[f'Y{i}'][:]
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    y_pred = model.predict(X)
    acc = accuracy(Y, y_pred)
    ent = entropy(Y, y_pred)
    df = pd.DataFrame({'Accuracy': acc, 'Entropy': ent})
    df.to_csv(args.out, index=False, sep='\t')


def main():
    parser = ArgumentParser()
    parser.add_argument("-model", help="model file")
    parser.add_argument("-weights", help="weights file")
    parser.add_argument("-dataset", help="test dataset file")
    parser.add_argument("-out", help="output file")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
