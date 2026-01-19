import os
import h5py
import pickle
import numpy as np
import pandas as pd
from typing import List
import logging
from itertools import count
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, Input, Cropping1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import tensorflow.keras.backend as kb
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
LOGGER.setLevel(logging.INFO)
HANDlER = logging.StreamHandler()
HANDlER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDlER)


def residual_block(filters: int, kernel_size: int, dilation_rate:int):
    """Residual block"""
    # Residual unit, 预激活结构B-A-C-B-A-C
    def unit(inputs):
        x = BatchNormalization()(inputs)
        x = Activation(activation='relu')(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same', kernel_initializer='he_uniform')(x)
        outputs = Add()([inputs, x])
        return outputs
    return unit


def spliceai(_filters: List[int], _kernel_size: List[int], _dilation_rate: List[int]) -> Model:
    """SpliceAI model"""
    # 计算上下文长度
    context_length = 2 * np.sum((np.array(_kernel_size) - 1) * np.array(_dilation_rate))
    # 限定输入形状4列
    inputs = Input(shape=(None, 4))
    # 32层空洞卷积残差神经网络，32个卷积核
    x = Conv1D(filters=32, kernel_size=1, dilation_rate=1, kernel_initializer='he_uniform')(inputs)
    short_cut = Conv1D(filters=32, kernel_size=1, dilation_rate=1, kernel_initializer='he_uniform')(x)
    for i in range(len(_filters)):
        x = residual_block(_filters[i], _kernel_size[i], _dilation_rate[i])(x)
        if (i + 1) % 4 == 0:
            _short_cut = Conv1D(filters=32, kernel_size=1, dilation_rate=1, kernel_initializer='he_uniform')(x)
            short_cut = Add()([short_cut, _short_cut])
    # 切掉上下文长度
    x = Cropping1D(cropping=(int(context_length / 2), int(context_length / 2)))(short_cut)
    outputs = Conv1D(filters=3, kernel_size=1, dilation_rate=1, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def categorical_crossentropy_2d(y_true, y_pred):
    # Standard categorical cross entropy for sequence outputs
    y_true = tf.cast(y_true, tf.float32)
    return - kb.mean(y_true[:, :, 0] * kb.log(y_pred[:, :, 0] + 1e-10)
                   + y_true[:, :, 1] * kb.log(y_pred[:, :, 1] + 1e-10)
                   + y_true[:, :, 2] * kb.log(y_pred[:, :, 2] + 1e-10))


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
    LOGGER.info(f'Total gene num : {len(genes)}. Total chunck num: {chunck_num}.')
    for i in range(chunck_num):
        if i + 1 == chunck_num:
            chunck_genes = genes[i * chunck_size:]
        else:
            chunck_genes = genes[i * chunck_size: (i + 1) * chunck_size]
        LOGGER.info(f'Chunck {i + 1} :')
        train_genes = chunck_genes[: -int(validation_split * chunck_size)]
        validation_genes = chunck_genes[-int(validation_split * chunck_size):]
        yield *concat_gene_data(h5f, train_genes), *concat_gene_data(h5f, validation_genes)


def cal_topk_accuracy(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred, average='macro', pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    idx_true = np.nonzero(y_true)[0]
    y_pred_sorted_idx = np.argsort(y_pred)
    threshold = y_pred[y_pred_sorted_idx[-idx_true.size]]
    if idx_true.size > 0:
        topk_accuracy = np.intersect1d(idx_true, y_pred_sorted_idx[-idx_true.size:]).size / idx_true.size
    else:
        topk_accuracy = 0
    return average_precision, roc_auc, topk_accuracy, idx_true.size, threshold


def run(args):
    # 创建模型
    filters = [32] * 4 + [32] * 4 + [32] * 4 + [32] * 4
    kernel_size = [11] * 4 + [11] * 4 + [21] * 4 + [41] * 4
    dilation_rate = [1] * 4 + [4] * 4 + [10] * 4 + [25] * 4
    model = spliceai(filters, kernel_size, dilation_rate)
    print(model.summary())
    # 模型检查点如果存在，则加载权重
    if os.path.exists(args.checkpoint):
        LOGGER.info(f'Load weights from {args.checkpoint}')
        model.load_weights(args.checkpoint)
    # 创建模型检查点
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    check_point = ModelCheckpoint(filepath=args.checkpoint, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, save_freq='epoch')
    batch_size = 32
    initial_lr = 0.001
    decay_rate = 0.8
    num = count()
    for epoch in range(10):
        h5f = h5py.File(args.dataset, 'r')
        metrics = []
        if epoch:
            lr = initial_lr * decay_rate
        else:
            lr = initial_lr
        LOGGER.info(f'Epoch {epoch + 1}, learning rate: {lr}')
        # 编译模型，优化器使用Adam，损失函数使用交叉熵，评估指标使用准确率
        model.compile(optimizer=Adam(learning_rate=lr), loss=categorical_crossentropy_2d, metrics=[TopKCategoricalAccuracy(k=1)])
        for x_train, y_train, x_valid, y_valid in generate_training_data(h5f):
            history = model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=10, validation_data=(x_valid, y_valid), validation_freq=1, callbacks=[early_stopping, check_point])
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
            _metrics = [
                *cal_topk_accuracy(y_train_acceptor, y_train_pred_acceptor),
                *cal_topk_accuracy(y_train_donor, y_train_pred_donor),
                *cal_topk_accuracy(y_valid_acceptor, y_valid_pred_acceptor),
                *cal_topk_accuracy(y_valid_donor, y_valid_pred_donor)]
            metrics.append(_metrics)
            LOGGER.info(f'Train & valid acc and donor metrics: {_metrics}')
            with open(os.path.join(args.out_dir, f'history{next(num)}.pkl'), 'wb') as f:
                pickle.dump(history.history, f)
        columns = [f'{k}_{j}_{i}' for i in ['train', 'valid'] for j in ['acceptor', 'donor'] for k in ['average_precision', 'roc_auc', 'topk_accuracy', 'true_size', 'threshold']]
        pd.DataFrame(metrics, columns=columns).to_csv(os.path.join(args.out_dir, f'metrics.{epoch}.tsv'), index=False, sep='\t')
        h5f.close()
    model.save(os.path.join(args.out_dir, args.out))


def main():
    parser = ArgumentParser()
    parser.add_argument('-checkpoint', help='checkpoint file')
    parser.add_argument('-dataset', help='dataset file')
    parser.add_argument('-out_dir', help='out dir')
    parser.add_argument('-out', help='out file')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
