from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, GRU
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
# from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
import numpy as np
import time

# 48-54 corresponding to 2014-2020
DATASET_INDEX = 48


MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True


def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = LSTM(128)(x)
    # x = GRU(128)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


if __name__ == "__main__":
    model = generate_model()
    pre_year = 2019
    year = 2015
    indice = "c"
    pre_weight_c = None
    pre_weight_m = None
    pre_weight_m_and_c = None
    accuracy_list = []
    # for i in range(5):

    train_start_time = time.time()
    if pre_weight_c is not None:
        # transfer learning with corn
        train_model(model, DATASET_INDEX, dataset_prefix='jd_stock_use_c' + str(year), epochs=600, batch_size=128, pre_weight=pre_weight_c)
        train_end_time = time.time()
        test_start_time = time.time()
        accuracy, loss = evaluate_model(model, DATASET_INDEX, dataset_prefix='jd_stock_use_c' + str(year), batch_size=128)
        test_end_time = time.time()
    elif pre_weight_m is not None:
        # transfer learning with meal
        train_model(model, DATASET_INDEX, dataset_prefix='jd_stock_use_m_' + str(year), epochs=600, batch_size=128, pre_weight=pre_weight_m)
        train_end_time = time.time()
        test_start_time = time.time()
        accuracy, loss = evaluate_model(model, DATASET_INDEX, dataset_prefix='jd_stock_use_m_' + str(year), batch_size=128)
        test_end_time = time.time()
    elif pre_weight_m_and_c is not None:
        # transfer learning with meal
        train_model(model, DATASET_INDEX, dataset_prefix='jd_stock_use_m_c_' + str(year), epochs=600, batch_size=128, pre_weight=pre_weight_m_and_c)
        train_end_time = time.time()
        test_start_time = time.time()
        accuracy, loss = evaluate_model(model, DATASET_INDEX, dataset_prefix='jd_stock_use_m_c_' + str(year), batch_size=128)
        test_end_time = time.time()
    else:
        # without transfer learning
        train_model(model, DATASET_INDEX, dataset_prefix=indice +'_stock_origin_' + str(year), epochs=600, batch_size=128, pre_weight=None)
        train_end_time = time.time()
        test_start_time = time.time()
        accuracy, loss = evaluate_model(model, DATASET_INDEX, dataset_prefix='indice'+'_stock_origin_' + str(year), batch_size=128)
        test_end_time = time.time()

