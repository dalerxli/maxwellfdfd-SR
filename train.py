#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import h5py
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import random
from model import *
import argparse
import copy
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler


model_name = input('SRCNN: S , VDSR: V , FSRCNN: F , FRSR: R , FRSR_L: L , FRSR_P: P , LapSRN: A :')

if model_name == 'S' or model_name == 'SRCNN' or model_name == 'srcnn' or model_name == 's':
    print('SRCNN')
    model_name = 'SRCNN'
    model_pick = 0
elif model_name == 'V' or model_name == 'VDSR' or model_name == 'vdsr' or model_name == 'v':
    print('VDSR')
    model_name = 'VDSR'
    model_pick = 1
elif model_name == 'F' or model_name == 'FSRCNN' or model_name == 'fsrcnn' or model_name == 'f':
    print('FSRCNN')
    model_name = 'FSRCNN'
    model_pick = 2
elif model_name == 'R' or model_name == 'FRSR' or model_name == 'fRsr' or model_name == 'r':
    print('FRSR')
    model_name = 'FRSR'
    model_pick = 3
elif model_name == 'L' or model_name == 'FRSR_L' or model_name == 'fRsr_l' or model_name == 'l':
    print('FRSR_L')
    model_name = 'FRSR_L'
    model_pick = 4
elif model_name == 'P' or model_name == 'FRSR_P' or model_name == 'fRsr_p' or model_name == 'p':
    print('FRSR_P')
    model_name = 'FRSR_P'
    model_pick = 5
elif model_name == 'A' or model_name == 'LapSRN' or model_name == 'lapsrn' or model_name == 'a':
    print('LapSRN')
    model_name = 'LapSRN'
    model_pick = 6
else:
    print('model_name error')

model_list = [SRCNN(), VDSR(), FSRCNN(), FRSR(), FRSR_L(), FRSR_P(), LapSRN()]

id = ['0001']
wavelength = []
for i in range(24):
    wavelength.append(400 + i*50)
# wavelength = 800
cs = [10, 5]
EH = ['x', 'y', 'z']

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--total_model_number", type=int, default=8)
parser.add_argument('--model', default='%s' %(model_name), help='Enter the dataset you want the model to train on')
opt = parser.parse_args()
print(opt)

def data(cs_value):
    x_path = []
    for i in range(len(image)):
        for j in range(len(wavelength)):
            path = '%s%s/%s/%s/x/' %(image[i], id[0], wavelength[j], cs_value)
            for (root, directories, files) in os.walk(path):
                for file in files:
                    if '.h5' in file:
                        file_path = os.path.join(root, file)
                        x_path.append(file_path)

    y_path = []
    for i in range(len(image)):
        for j in range(len(wavelength)):
            path = '%s%s/%s/%s/y/' % (image[i], id[0], wavelength[j], cs_value)
            for (root, directories, files) in os.walk(path):
                for file in files:
                    if '.h5' in file:
                        file_path = os.path.join(root, file)
                        y_path.append(file_path)

    z_path = []
    for i in range(len(image)):
        for j in range(len(wavelength)):
            path = '%s%s/%s/%s/z/' % (image[i], id[0], wavelength[j], cs_value)
            for (root, directories, files) in os.walk(path):
                for file in files:
                    if '.h5' in file:
                        file_path = os.path.join(root, file)
                        z_path.append(file_path)


    return x_path, y_path, z_path

def h5_data(k):
    with h5py.File(k, 'r') as f:
        a_group_key = list(f.keys())[0]
        k = list(f[a_group_key])
    k = np.array(k)
    return k

def h5_XYZ(X_matrix, Y_matrix):
    Y = []
    for i in range(len(Y_matrix[0])):
        Y.append(np.stack([h5_data(Y_matrix[0][i])[220:620],
                              h5_data(Y_matrix[1][i])[220:620],
                              h5_data(Y_matrix[2][i])[220:620]], axis=2))

    X = []
    for i in range(len(Y_matrix[0])):
        X.append(np.stack([h5_data(X_matrix[0][i])[110:310],
                               h5_data(X_matrix[1][i])[110:310],
                               h5_data(X_matrix[2][i])[110:310]], axis=2))

    return X, Y

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def divide_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def random_data(x, random_list):
    x_path, y_path, z_path = [], [], []
    for i in random_list:
        x_path.append(x[0][i])
        y_path.append(x[1][i])
        z_path.append(x[2][i])
    return x_path, y_path, z_path

def save_3():
    data_all_1 = np.hstack((np.array(np.reshape(X_RMSE_train_all, (-1, 1))), np.array(np.reshape(X_R2_train_all, (-1, 1))),
                          np.array(np.reshape(Y_RMSE_train_all, (-1, 1))), np.array(np.reshape(Y_R2_train_all, (-1, 1))),
                          np.array(np.reshape(Z_RMSE_train_all, (-1, 1))), np.array(np.reshape(Z_R2_train_all, (-1, 1))),
                          np.array(np.reshape(X_RMSE_test_all, (-1, 1))), np.array(np.reshape(X_R2_test_all, (-1, 1))),
                          np.array(np.reshape(Y_RMSE_test_all, (-1, 1))), np.array(np.reshape(Y_R2_test_all, (-1, 1))),
                          np.array(np.reshape(Z_RMSE_test_all, (-1, 1))), np.array(np.reshape(Z_R2_test_all, (-1, 1))),
                          np.array(np.reshape(model_time, (-1, 1)))))
    pd.DataFrame(data_all_1).to_csv('%s/train_3_result.csv' % (train_save_path),
                                  header=['X_RMSE_train', 'X_R2_train', 'Y_RMSE_train', 'Y_R2_train', 'Z_RMSE_train', 'Z_R2_train',
                                          'X_RMSE_test', 'X_R2_test', 'Y_RMSE_test', 'Y_R2_test', 'Z_RMSE_test', 'Z_R2_test'
                                      ,'time'], index=False)
    return print("save")

def evaluation_train(k):
    RMSE = mean_squared_error(train_result[j][:, :, k], y_train_real[j][:, :, k]) ** 0.5
    R2 = r2_score(train_result[j][:, :, k], y_train_real[j][:, :, k])
    return RMSE, R2

def evaluation_test(k):
    RMSE = mean_squared_error(test_result[j][:, :, k], y_test_real[j][:, :, k]) ** 0.5
    R2 = r2_score(test_result[j][:, :, k], y_test_real[j][:, :, k])
    return RMSE, R2

def evaluation_all(RMSE, R2):
    RMSE_ = np.sum(RMSE) / np.array(RMSE).shape
    R2_ = np.sum(R2) / np.array(R2).shape
    return RMSE_, R2_

data_location = input('server --> s or computer --> c: ')
if data_location == 's':
    print('server')
    # server
    random_image = '/home/jang/project/SRmodel/data/random_jangwon/'
    simple_image = '/home/jang/project/SRmodel/data/simple_jangwon/'
    image = [simple_image, random_image]
    save_path = createFolder('/home/jang/project/SRmodel/result/%s/' % (model_name))
elif data_location == 'c':
    print('computer')
    # computer
    random_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/random_jangwon/'
    simple_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/simple_jangwon/'
    image = [simple_image, random_image]
    save_path = createFolder('D:/project/SR/SRmodel/result_practice/%s/' % (model_name))
else:
    print('data location error')

model_save_path = createFolder('%s/save_model/' %(save_path))
train_save_path = createFolder('%s/train/'%(save_path))

xdata_file = data(10)
ydata_file = data(5)
random_set = random.sample(range(0, len(xdata_file[0])), len(xdata_file[0]))
random_choice = list(divide_list(random_set, 1200))

model_time, train_acc, test_acc = [], [], []

X_RMSE_train_all, X_R2_train_all, X_RMSE_test_all, X_R2_test_all = [], [], [], []
Y_RMSE_train_all, Y_R2_train_all, Y_RMSE_test_all, Y_R2_test_all = [], [], [], []
Z_RMSE_train_all, Z_R2_train_all, Z_RMSE_test_all, Z_R2_test_all = [], [], [], []

for model_number in range(opt.total_model_number):

    X_RMSE_train, X_R2_train, X_RMSE_test, X_R2_test = [], [], [], []
    Y_RMSE_train, Y_R2_train, Y_RMSE_test, Y_R2_test = [], [], [], []
    Z_RMSE_train, Z_R2_train, Z_RMSE_test, Z_R2_test = [], [], [], []

    print('\nmodel number: %s' %(model_number+1))
    train_choice = random_choice[model_number][:int(len(random_choice[model_number])*0.8)]
    test_choice = random_choice[model_number][int(len(random_choice[model_number]) * 0.8):]
    X_train_i = random_data(xdata_file, train_choice)
    X_test_i = random_data(xdata_file, test_choice)
    y_train_i = random_data(ydata_file, train_choice)
    y_test_i = random_data(ydata_file, test_choice)

    X_train, y_train = h5_XYZ(X_train_i, y_train_i)
    X_test, y_test = h5_XYZ(X_test_i, y_test_i)

    # (n, input shape, input shape, channels)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    scalers_x = StandardScaler()
    scalers_y = StandardScaler()
    scalers_z = StandardScaler()

    scalers_x.fit(X_train[:, :, :, 0].reshape(-1, 1))
    scalers_y.fit(X_train[:, :, :, 1].reshape(-1, 1))
    scalers_z.fit(X_train[:, :, :, 2].reshape(-1, 1))

    X_train[:, :, :, 0:1] = scalers_x.transform(X_train[:, :, :, 0].reshape(-1, 1)).reshape(X_train.shape[0],
                                                                                            X_train.shape[1],
                                                                                            X_train.shape[2], 1)
    X_train[:, :, :, 1:2] = scalers_y.transform(X_train[:, :, :, 1].reshape(-1, 1)).reshape(X_train.shape[0],
                                                                                            X_train.shape[1],
                                                                                            X_train.shape[2], 1)
    X_train[:, :, :, 2:3] = scalers_z.transform(X_train[:, :, :, 2].reshape(-1, 1)).reshape(X_train.shape[0],
                                                                                            X_train.shape[1],
                                                                                            X_train.shape[2], 1)

    X_test[:, :, :, 0:1] = scalers_x.transform(X_test[:, :, :, 0].reshape(-1, 1)).reshape(X_test.shape[0],
                                                                                          X_test.shape[1],
                                                                                          X_test.shape[2], 1)
    X_test[:, :, :, 1:2] = scalers_y.transform(X_test[:, :, :, 1].reshape(-1, 1)).reshape(X_test.shape[0],
                                                                                          X_test.shape[1],
                                                                                          X_test.shape[2], 1)
    X_test[:, :, :, 2:3] = scalers_z.transform(X_test[:, :, :, 2].reshape(-1, 1)).reshape(X_test.shape[0],
                                                                                          X_test.shape[1],
                                                                                          X_test.shape[2], 1)
    y_train_real = copy.deepcopy(y_train)
    y_test_real = copy.deepcopy(y_test)

    y_train[:, :, :, 0:1] = scalers_x.transform(y_train[:, :, :, 0].reshape(-1, 1)).reshape(y_train.shape[0],
                                                                                            y_train.shape[1],
                                                                                            y_train.shape[2], 1)
    y_train[:, :, :, 1:2] = scalers_y.transform(y_train[:, :, :, 1].reshape(-1, 1)).reshape(y_train.shape[0],
                                                                                            y_train.shape[1],
                                                                                            y_train.shape[2], 1)
    y_train[:, :, :, 2:3] = scalers_z.transform(y_train[:, :, :, 2].reshape(-1, 1)).reshape(y_train.shape[0],
                                                                                            y_train.shape[1],
                                                                                            y_train.shape[2], 1)

    y_test[:, :, :, 0:1] = scalers_x.transform(y_test[:, :, :, 0].reshape(-1, 1)).reshape(y_test.shape[0],
                                                                                          y_test.shape[1],
                                                                                          y_test.shape[2], 1)
    y_test[:, :, :, 1:2] = scalers_y.transform(y_test[:, :, :, 1].reshape(-1, 1)).reshape(y_test.shape[0],
                                                                                          y_test.shape[1],
                                                                                          y_test.shape[2], 1)
    y_test[:, :, :, 2:3] = scalers_z.transform(y_test[:, :, :, 2].reshape(-1, 1)).reshape(y_test.shape[0],
                                                                                          y_test.shape[1],
                                                                                          y_test.shape[2], 1)

    if os.path.exists('%s/%s.h5' % (model_save_path, model_number + 1)) == False:
        model = model_list[model_pick]
        if model_number == 0:
            model.summary()

        start_time = time.time()
        checkpoint = ModelCheckpoint('%s/%s.h5' % (model_save_path, model_number + 1), monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
        history = model.fit(X_train, y_train, batch_size=opt.batch_size, epochs=opt.epochs, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])
        # model.save('%s/%s.h5' % (model_save_path, model_number + 1))
    else:
        start_time = time.time()
        model = load_model(filepath='%s/%s.h5' % (model_save_path, model_number + 1))
        print('model %s exist' % (model_number + 1))


    train_acc.append(model.evaluate(X_train, y_train, verbose=0))
    test_acc.append(model.evaluate(X_test, y_test, verbose=0))
    train_result = model.predict(X_train, verbose=0)
    test_result = model.predict(X_test, verbose=0)
    print("--- %s seconds ---" % (time.time() - start_time))
    model_time.append(time.time() - start_time)

    train_result[:, :, :, 0:1] = scalers_x.inverse_transform(train_result[:, :, :, 0].reshape(-1, 1)).reshape(train_result.shape[0],
                                                                                                              train_result.shape[1],
                                                                                                              train_result.shape[2], 1)
    train_result[:, :, :, 1:2] = scalers_y.inverse_transform(train_result[:, :, :, 1].reshape(-1, 1)).reshape(train_result.shape[0],
                                                                                                              train_result.shape[1],
                                                                                                              train_result.shape[2], 1)
    train_result[:, :, :, 2:3] = scalers_z.inverse_transform(train_result[:, :, :, 2].reshape(-1, 1)).reshape(train_result.shape[0],
                                                                                                              train_result.shape[1],
                                                                                                              train_result.shape[2], 1)

    test_result[:, :, :, 0:1] = scalers_x.inverse_transform(test_result[:, :, :, 0].reshape(-1, 1)).reshape(test_result.shape[0],
                                                                                                            test_result.shape[1],
                                                                                                            test_result.shape[2], 1)
    test_result[:, :, :, 1:2] = scalers_y.inverse_transform(test_result[:, :, :, 1].reshape(-1, 1)).reshape(test_result.shape[0],
                                                                                                            test_result.shape[1],
                                                                                                            test_result.shape[2], 1)
    test_result[:, :, :, 2:3] = scalers_z.inverse_transform(test_result[:, :, :, 2].reshape(-1, 1)).reshape(test_result.shape[0],
                                                                                                            test_result.shape[1],
                                                                                                            test_result.shape[2], 1)

    train_result[train_result < 0] = 0
    test_result[test_result < 0] = 0

    for j in range(np.array(X_train).shape[0]):
        X_RMSE_train.append(evaluation_train(0)[0])
        X_R2_train.append(evaluation_train(0)[1])

        Y_RMSE_train.append(evaluation_train(1)[0])
        Y_R2_train.append(evaluation_train(1)[1])

        Z_RMSE_train.append(evaluation_train(2)[0])
        Z_R2_train.append(evaluation_train(2)[1])

    for j in range(np.array(X_test).shape[0]):
        X_RMSE_test.append(evaluation_test(0)[0])
        X_R2_test.append(evaluation_test(0)[1])

        Y_RMSE_test.append(evaluation_test(1)[0])
        Y_R2_test.append(evaluation_test(1)[1])

        Z_RMSE_test.append(evaluation_test(2)[0])
        Z_R2_test.append(evaluation_test(2)[1])

    X_RMSE_train_all, X_R2_train_all = evaluation_all(X_RMSE_train, X_R2_train)
    Y_RMSE_train_all, Y_R2_train_all = evaluation_all(Y_RMSE_train, Y_R2_train)
    Z_RMSE_train_all, Z_R2_train_all = evaluation_all(Z_RMSE_train, Z_R2_train)

    X_RMSE_test_all, X_R2_test_all = evaluation_all(X_RMSE_test, X_R2_test)
    Y_RMSE_test_all, Y_R2_test_all = evaluation_all(Y_RMSE_test, Y_R2_test)
    Z_RMSE_test_all, Z_R2_test_all = evaluation_all(Z_RMSE_test, Z_R2_test)

    print('train error')
    print(
        'X_RMSE_train: %.4f, X_R2_train: %.4f, Y_RMSE_train: %.4f, Y_R2_train: %.4f, Z_RMSE_train: %.4f, Z_R2_train: %.4f'
        % (X_RMSE_train_all, X_R2_train_all,
           Y_RMSE_train_all, Y_R2_train_all,
           Z_RMSE_train_all, Z_R2_train_all))
    print('test error')
    print(
        'X_RMSE_test: %.4f, X_R2_test: %.4f, Y_RMSE_test: %.4f, Y_R2_test: %.4f, Z_RMSE_test: %.4f, Z_R2_test: %.4f'
        % (X_RMSE_test_all, X_R2_test_all,
           Y_RMSE_test_all, Y_R2_test_all,
           Z_RMSE_test_all, Z_R2_test_all))

data_all = np.hstack((np.array(train_acc).reshape(-1, 2), np.array(test_acc).reshape(-1, 2), np.array(model_time).reshape(-1, 1)))
pd.DataFrame(data_all).to_csv('%s/train_result.csv' %(train_save_path),
                              header=['Train loss', 'Train mse', 'Test loss', 'Test mse', 'time'], index=False)

save_3()