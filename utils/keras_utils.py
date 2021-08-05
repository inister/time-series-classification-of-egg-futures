import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-paper')
# mpl.use('TKAgg')

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.models import Model
from keras.layers import Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Permute
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras import backend as K

from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
    cutoff_sequence
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST


def multi_label_log_loss(y_pred, y_true):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def _average_gradient_norm(model, X_train, y_train, batch_size):
    # just checking if the model was already compiled
    if not hasattr(model, "train_function"):
        raise RuntimeError("You must compile your model before using it.")

    weights = model.trainable_weights  # weight tensors

    get_gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors

    input_tensors = [
        # input data
        model.inputs[0],
        # how much to weight each sample by
        model.sample_weights[0],
        # labels
        model.targets[0],
        # train or test mode
        K.learning_phase()
    ]

    grad_fct = K.function(inputs=input_tensors, outputs=get_gradients)

    steps = 0
    total_norm = 0
    s_w = None

    nb_steps = X_train.shape[0] // batch_size

    if X_train.shape[0] % batch_size == 0:
        pad_last = False
    else:
        pad_last = True

    def generator(X_train, y_train, pad_last):
        for i in range(nb_steps):
            X = X_train[i * batch_size: (i + 1) * batch_size, ...]
            y = y_train[i * batch_size: (i + 1) * batch_size, ...]

            yield (X, y)

        if pad_last:
            X = X_train[nb_steps * batch_size:, ...]
            y = y_train[nb_steps * batch_size:, ...]

            yield (X, y)

    datagen = generator(X_train, y_train, pad_last)

    while steps < nb_steps:
        X, y = next(datagen)
        # set sample weights to one
        # for every input
        if s_w is None:
            s_w = np.ones(X.shape[0])

        gradients = grad_fct([X, s_w, y, 0])
        total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
        steps += 1

    if pad_last:
        X, y = next(datagen)
        # set sample weights to one
        # for every input
        if s_w is None:
            s_w = np.ones(X.shape[0])

        gradients = grad_fct([X, s_w, y, 0])
        total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
        steps += 1

    return total_norm / float(steps)


def draw_acc_loss_graph(history):
    # val_loss,val_acc,loss,acc
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    acc = history.history['accuracy']
    print("val_loss =", val_loss)
    print("val_acc =", val_acc)
    print("loss =", loss)
    print("acc =", acc)
    epoch = range(1, len(acc) + 1)

    plt.subplot(121)
    plt.plot(epoch, acc, 'red', label='Training_acc')
    plt.plot(epoch, val_acc, 'blue', label="Validation acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(122)
    plt.plot(epoch, loss, 'red', label='Training loss')
    plt.plot(epoch, val_loss, 'blue', label="Validation loss")
    plt.title("Loss")
    plt.legend()
    plt.show()
    plt.savefig("./images/jd_2019.png", dpi=300)


# keras.model 学习资料：https://keras.io/zh/models/model/
def train_model(model: Model, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128,
                val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3, monitor='loss', optimization_mode='auto',
                compile_model=True, pre_weight=None):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    # 计算最大时间步数，最大变量数
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)

    classes = np.unique(y_train)
    # LabelEncoder: 将类别数据数字化的方法
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    # np.bincount该函数就是将原来数组 x 中的每一项出现的频次记录下来，存放到 y 中。
    recip_freq = len(y_train) / (len(le.classes_) *
                                 np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    # 将类向量转化为二进制类矩阵
    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s_weights.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s_fold_%d_weights.h5" % (dataset_prefix, dataset_fold_id)

    # add pre weights
    if pre_weight is not None:
        model.load_weights(pre_weight)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))
    draw_acc_loss_graph(history)


def evaluate_model(model: Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128,
                   test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    # X_test, y_test, _, _, is_timeseries = load_dataset_at(dataset_id,
    #                                                       fold_index=dataset_fold_id,
    #                                                       normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, max_nb_variables)

    if not is_timeseries:
        # keras只能接受长度相等的序列输入。当我们的数据集中出现了长度不等的序列时，可以使用pad_sequence()函数将序列转化为经过填充以后得到的一个长度相同新的序列。
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    # metrics包括['accuracy', 'mse', 'mae'] model.compile用于配置训练模型
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if dataset_fold_id is None:
        weight_fn = "./weights/%s_weights.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s_fold_%d_weights.h5" % (dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    # 在测试模式下返回模型的误差值和评估标准值。
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    # 输出预测的类概率
    # probabilities = model.predict(X_test, batch_size=batch_size)
    # print("probabilities: ", probabilities)
    print()
    print("Final Accuracy : ", accuracy)

    return accuracy, loss


def set_trainable(layer, value):
    layer.trainable = value

    # case: container
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            set_trainable(l, value)

    # case: wrapper (which is a case not covered by the PR)
    if hasattr(layer, 'layer'):
        set_trainable(layer.layer, value)


def compute_average_gradient_norm(model: Model, dataset_id, dataset_fold_id=None, batch_size=128,
                                  cutoff=None, normalize_timeseries=False, learning_rate=1e-3):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    max_timesteps, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    y_train = to_categorical(y_train, len(np.unique(y_train)))

    optm = Adam(lr=learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    average_gradient = _average_gradient_norm(model, X_train, y_train, batch_size)
    print("Average gradient norm : ", average_gradient)


class MaskablePermute(Permute):

    def __init__(self, dims, **kwargs):
        super(MaskablePermute, self).__init__(dims, **kwargs)
        self.supports_masking = True


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall))
