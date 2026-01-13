import sys
import h5py
from typing import List
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras.models import load_model


def concat_gene_data(h5f: h5py.File, genes: List[str]):
    """将基因编码后的数据进行合并"""
    X, Y = [], []
    for gene in genes:
        X.append(h5f[f'X/{gene}'][:])
        Y.append(h5f[f'Y/{gene}'][:])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    return X, Y


def generate_training_data(h5f: h5py.File):
    """创建训练数据，将基因分块，并划分训练集和验证集"""
    chunck_size = 100
    validation_split = 0.1
    genes = list(h5f['X'].keys())
    # 随机打乱基因顺序
    np.random.shuffle(genes)
    chunck_num = len(genes) // chunck_size
    for i in range(chunck_num):
        if i + 1 == chunck_num:
            chunck_genes = genes[i * chunck_size:]
        else:
            chunck_genes = genes[i * chunck_size: (i + 1) * chunck_size]
        train_genes = chunck_genes[: -int(validation_split * chunck_size)]
        validation_genes = chunck_genes[-int(validation_split * chunck_size):]
        yield *concat_gene_data(h5f, train_genes), *concat_gene_data(h5f, validation_genes)


def cal_topk_accuracy(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred, average='macro', pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    idx_true = np.nonzero(y_true)[0]
    y_pred_sorted = np.sort(y_pred)
    y_pred_sorted_idx = np.argsort(y_pred)
    threshold = y_pred[y_pred_sorted_idx[-idx_true.size]]
    if idx_true.size > 0:
        topk_accuracy = np.intersect1d(idx_true, y_pred_sorted_idx[-idx_true.size:]).size / idx_true.size
    else:
        topk_accuracy = 0
    return average_precision, roc_auc, topk_accuracy, idx_true.size, threshold


def main():
    model = load_model(sys.argv[1])
    h5f = h5py.File(sys.argv[2], 'r')
    for x_train, y_train, x_valid, y_valid in generate_training_data(h5f):
        y_train_pred = model.predict(x_train)
        y_train_pred_acceptor = y_train_pred[:, :, 1].flatten()
        y_train_pred_donor = y_train_pred[:, :, 2].flatten()
        y_train_acceptor = y_train[:, :, 1].flatten()
        y_train_donor = y_train[:, :, 2].flatten()
        y_valid_pred = model.predict(x_valid)
        y_valid_pred_acceptor = y_valid_pred[:, :, 1].flatten()
        y_valid_pred_donor = y_valid_pred[:, :, 2].flatten()
        y_valid_acceptor = y_valid[:, :, 1].flatten()
        y_valid_donor = y_valid[:, :, 2].flatten()
        print(f'Train acceptor metrics: {cal_topk_accuracy(y_train_acceptor, y_train_pred_acceptor)}')
        print(f'Train donor metrics: {cal_topk_accuracy(y_train_donor, y_train_pred_donor)}')
        print(f'Valid acceptor metrics: {cal_topk_accuracy(y_valid_acceptor, y_valid_pred_acceptor)}')
        print(f'Valid donor metrics: {cal_topk_accuracy(y_valid_donor, y_valid_pred_donor)}')
    h5f.close()


if __name__ == '__main__':
    main()
