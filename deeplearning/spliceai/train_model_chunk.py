import os
import h5py
import pickle
import numpy as np
from argparse import ArgumentParser
from itertools import count
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, Input, Cropping1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def residual_block(filters, kernel_size, dilation_rate):
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


def spliceai(_filters, _kernel_size, _dilation_rate) -> Model:
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


def run(args):
    filters = [32] * 4 + [32] * 4 + [32] * 4 + [32] * 4
    kernel_size = [11] * 4 + [11] * 4 + [21] * 4 + [41] * 4
    dilation_rate = [1] * 4 + [4] * 4 + [10] * 4 + [25] * 4
    model = spliceai(filters, kernel_size, dilation_rate)
    print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    if os.path.exists(args.checkpoint):
        model.load_weights(args.checkpoint)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
    check_point = ModelCheckpoint(filepath=args.checkpoint, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=True, save_freq='epoch')
    # 数据集分成batch_num份，batch大小为32, 并进行训练
    h5f = h5py.File(args.dataset, 'r')
    batch_size = 32
    total = len(h5f.keys())
    batch_num = 20
    batch_datasize = total // batch_num
    counter = count()
    batch_now = next(counter)
    X, Y = [], []
    for i in range(total):
        X.append(h5f[f'X{i}'][:])
        Y.append(h5f[f'Y{i}'][:])
        if (i + 1) % batch_datasize == 0 and (batch_now + 1) < batch_num:
            X = np.array(X)
            Y = np.array(Y)
            print(f'Batch {batch_now + 1} / {batch_num}')
            print(f'X shape: {X.shape}, Y shape: {Y.shape}')
            history = model.fit(X, Y, batch_size=batch_size, verbose=2, epochs=10, validation_split=0.1, shuffle=True, callbacks=[early_stopping, check_point])
            with open(os.path.join(args.out_dir, f'history_{batch_now + 1}.pkl'), 'wb') as f:
                pickle.dump(history.history, f)
            batch_now = next(counter)
            X, Y = [], []
        elif (batch_now + 1) == batch_num:
            pass
    X = np.array(X)
    Y = np.array(Y)
    print(f'Batch {batch_now + 1} / {batch_num}')
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    history = model.fit(X, Y, batch_size=batch_size, verbose=2, epochs=10, validation_split=0.1, shuffle=True, callbacks=[early_stopping, check_point])
    with open(os.path.join(args.out_dir, f'history_{batch_now + 1}.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
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
