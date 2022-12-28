import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2DTranspose, Activation, add, Conv2D, PReLU, LeakyReLU
import tensorflow.keras.backend as K
import numpy as np

def SRCNN():
    # SRCNN Model
    input_img = Input(shape=(200, 200, 3))
    input_img_ip = tf.image.resize(input_img, [400, 400], method='bicubic')
    model = Conv2D(64, (9, 9), padding='same', kernel_initializer='he_normal')(input_img_ip)
    model = Activation('relu')(model)
    model = Conv2D(32, (9, 9), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(3, (5, 5), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('linear')(model)
    output_img = model
    model = Model(input_img, output_img)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model

def FSRCNN():
    # FSRCNN Model
    input_img = Input(shape=(200, 200, 3))
    model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = PReLU()(model)
    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2DTranspose(3, (9, 9), strides=(2, 2), padding='same')(model)
    output_img = model
    model = Model(input_img, output_img)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model

def VDSR():
    # VDSR model
    input_img = Input(shape=(200, 200, 3))
    input_img_ip = tf.image.resize(input_img, [400, 400], method='bicubic')
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img_ip)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model
    output_img = add([res_img, input_img_ip])
    model = Model(input_img, output_img)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model

def FRSR():
    input_img = Input(shape=(200, 200, 3))
    input_img_ip = tf.image.resize(input_img, [400, 400], method='bicubic')
    model = Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = Activation('relu')(model)
    model = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(3, (9, 9), strides=(2, 2), padding='same')(model)
    res_img = model
    output_img = add([res_img, input_img_ip])
    model = Model(input_img, output_img)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model

def FRSR_L():
    input_img = Input(shape=(200, 200, 3))
    input_img_ip = tf.image.resize(input_img, [400, 400], method='bicubic')
    model = Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = LeakyReLU()(model)
    model = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2DTranspose(3, (9, 9), strides=(2, 2), padding='same')(model)
    res_img = model
    output_img = add([res_img, input_img_ip])
    model = Model(input_img, output_img)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()
    return model

def FRSR_P():
    input_img = Input(shape=(200, 200, 3))
    input_img_ip = tf.image.resize(input_img, [400, 400], method='bicubic')
    model = Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = PReLU()(model)
    model = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2DTranspose(3, (9, 9), strides=(2, 2), padding='same')(model)
    res_img = model
    output_img = add([res_img, input_img_ip])
    model = Model(input_img, output_img)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model


def L1_Charbonnier_loss(y_true, y_pred):
    """L1 Charbonnierloss."""
    eps = 1e-6
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    diff = y_true-y_pred
    error = K.sqrt( diff * diff + eps )
    loss = K.sum(error)
    return loss

def LapSRN():
    # LapSRN Model
    input_img = Input(shape=(200, 200, 3))
    model = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model = LeakyReLU()(model)
    model = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    model = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU()(model)
    embedding = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model)

    # upsampling
    model_up = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model_up = LeakyReLU()(model_up)
    model_up = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_up)
    model_up = LeakyReLU()(model_up)
    upsample = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(model_up)

    # residual
    model_res = LeakyReLU()(embedding)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = add([model_res, embedding])
    model_res = LeakyReLU()(model_res)
    model_res = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)
    model_res = LeakyReLU()(model_res)
    model_res = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_normal')(model_res)

    output_img = add([upsample, model_res])
    model = Model(input_img, output_img)

    adam = Adam(lr=0.0003)
    # model.compile(optimizer=adam, loss=L1_Charbonnier_loss, metrics=['accuracy'])
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model