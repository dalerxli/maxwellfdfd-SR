import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2DTranspose, Activation, add, Conv2D, PReLU, LeakyReLU

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

def RFSR():
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

def RFSR_L():
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
    return model

def RFSR_P():
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