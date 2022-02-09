#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import scipy.interpolate
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import random
import matplotlib.pyplot as plt


id = ['0001']
wavelength = []
for i in range(24):
    wavelength.append(400 + i*50)
# wavelength = 800
cs = [10, 5]
EH = ['x', 'y', 'z']

def interpolation_matrix(X, Y):
    Source = X
    x = np.arange(0, Source.shape[0])
    fit_x = scipy.interpolate.interp1d(x, Source, axis=0, kind='cubic')
    Target = fit_x(np.linspace(0, Source.shape[0] - 1, Y.shape[0]))

    y = np.arange(0, Target.shape[1])
    fit_y = scipy.interpolate.interp1d(y, Target, axis=1, kind='cubic')
    Target = fit_y(np.linspace(0, Target.shape[1] - 1, Y.shape[1]))

    return Target

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
    HR = h5_data(Y_matrix[0][0])[220:620]
    for i in range(len(Y_matrix[0])):
        X.append(np.stack([interpolation_matrix(h5_data(X_matrix[0][i])[110:310], HR),
                            interpolation_matrix(h5_data(X_matrix[1][i])[110:310], HR),
                            interpolation_matrix(h5_data(X_matrix[2][i])[110:310], HR)],
                     axis=2))

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
                          np.array(np.reshape(Z_RMSE_test_all, (-1, 1))), np.array(np.reshape(Z_R2_test_all, (-1, 1)))))
    pd.DataFrame(data_all_1).to_csv('%s/train_3_result.csv' % (train_save_path),
                                  header=['X_RMSE_train', 'X_R2_train', 'Y_RMSE_train', 'Y_R2_train', 'Z_RMSE_train', 'Z_R2_train',
                                          'X_RMSE_test', 'X_R2_test', 'Y_RMSE_test', 'Y_R2_test', 'Z_RMSE_test', 'Z_R2_test'], index=False)
    return print("save")

data_location = input('server --> s or computer --> c: ')
if data_location == 's':
    print('server')
    # server
    random_image = '/home/jang/project/SRmodel/data/random_jangwon/'
    simple_image = '/home/jang/project/SRmodel/data/simple_jangwon/'
    image = [simple_image, random_image]
elif data_location == 'c':
    print('computer')
    # computer
    random_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/random_jangwon/'
    simple_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/simple_jangwon/'
    image = [simple_image, random_image]
else:
    print('data location error')

if data_location == 's':
    # server
    save_path = createFolder('/home/jang/project/SRmodel/result/interpolation/')
elif data_location == 'c':
    # computer
    save_path = createFolder('D:/project/SR/SRmodel/result/interpolation/')
else:
    print('data location error')

train_save_path = createFolder('%s/train/'%(save_path))

xdata_file = data(10)
ydata_file = data(5)
random_set = random.sample(range(0, len(xdata_file[0])), len(xdata_file[0]))
random_choice = list(divide_list(random_set, 1200))

model_time, train_acc, test_acc = [], [], []

X_RMSE_train, X_R2_train, X_RMSE_test, X_R2_test = [], [], [], []
Y_RMSE_train, Y_R2_train, Y_RMSE_test, Y_R2_test = [], [], [], []
Z_RMSE_train, Z_R2_train, Z_RMSE_test, Z_R2_test = [], [], [], []
X_RMSE_train_all, X_R2_train_all, X_RMSE_test_all, X_R2_test_all = [], [], [], []
Y_RMSE_train_all, Y_R2_train_all, Y_RMSE_test_all, Y_R2_test_all = [], [], [], []
Z_RMSE_train_all, Z_R2_train_all, Z_RMSE_test_all, Z_R2_test_all = [], [], [], []

plt.figure(figsize=(8, 8))

for model_number in range(8):
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

    X_train[X_train < 0] = 0
    y_train[y_train < 0] = 0
    X_test[X_test < 0] = 0
    y_test[y_test < 0] = 0

    for j in range(np.array(X_train).shape[0]):
        X_RMSE_train.append(mean_squared_error(X_train[j][:, :, 0],
                                               y_train[j][:, :, 0]) ** 0.5)
        X_R2_train.append(r2_score(X_train[j][:, :, 0],
                                   y_train[j][:, :, 0]))
        Y_RMSE_train.append(mean_squared_error(X_train[j][:, :, 1],
                                               y_train[j][:, :, 1]) ** 0.5)
        Y_R2_train.append(r2_score(X_train[j][:, :, 1],
                                   y_train[j][:, :, 1]))
        Z_RMSE_train.append(mean_squared_error(X_train[j][:, :, 2],
                                               y_train[j][:, :, 2]) ** 0.5)
        Z_R2_train.append(r2_score(X_train[j][:, :, 2],
                                   y_train[j][:, :, 2]))

    for j in range(np.array(X_test).shape[0]):
        X_RMSE_test.append(mean_squared_error(X_test[j][:, :, 0],
                                              y_test[j][:, :, 0]) ** 0.5)
        X_R2_test.append(r2_score(X_test[j][:, :, 0],
                                  y_test[j][:, :, 0]))
        Y_RMSE_test.append(mean_squared_error(X_test[j][:, :, 1],
                                              y_test[j][:, :, 1]) ** 0.5)
        Y_R2_test.append(r2_score(X_test[j][:, :, 1],
                                  y_test[j][:, :, 1]))
        Z_RMSE_test.append(mean_squared_error(X_test[j][:, :, 2],
                                              y_test[j][:, :, 2]) ** 0.5)
        Z_R2_test.append(r2_score(X_test[j][:, :, 2],
                                  y_test[j][:, :, 2]))
        if j ==1:
            plt.scatter(X_test[j][:, :, 0].reshape(1, -1), y_test[j][:, :, 0].reshape(1, -1), c='black', s=4, alpha=0.4)
            plt.scatter(X_test[j][:, :, 1].reshape(1, -1), y_test[j][:, :, 1].reshape(1, -1), c='black', s=4, alpha=0.4)
            plt.scatter(X_test[j][:, :, 2].reshape(1, -1),  y_test[j][:, :, 2].reshape(1, -1), c='black', s=4, alpha=0.4)

    X_RMSE_train_all.append(np.sum(X_RMSE_train) / np.array(X_RMSE_train).shape)
    X_R2_train_all.append(np.sum(X_R2_train) / np.array(X_R2_train).shape)
    X_RMSE_test_all.append(np.sum(X_RMSE_test) / np.array(X_RMSE_test).shape)
    X_R2_test_all.append(np.sum(X_R2_test) / np.array(X_R2_test).shape)

    Y_RMSE_train_all.append(np.sum(Y_RMSE_train) / np.array(Y_RMSE_train).shape)
    Y_R2_train_all.append(np.sum(Y_R2_train) / np.array(Y_R2_train).shape)
    Y_RMSE_test_all.append(np.sum(Y_RMSE_test) / np.array(Y_RMSE_test).shape)
    Y_R2_test_all.append(np.sum(Y_R2_test) / np.array(Y_R2_test).shape)

    Z_RMSE_train_all.append(np.sum(Z_RMSE_train) / np.array(Z_RMSE_train).shape)
    Z_R2_train_all.append(np.sum(Z_R2_train) / np.array(Z_R2_train).shape)
    Z_RMSE_test_all.append(np.sum(Z_RMSE_test) / np.array(Z_RMSE_test).shape)
    Z_R2_test_all.append(np.sum(Z_R2_test) / np.array(Z_R2_test).shape)

    print('train error')
    print(
        'X_RMSE_train: %.4f, X_R2_train: %.4f, Y_RMSE_train: %.4f, Y_R2_train: %.4f, Z_RMSE_train: %.4f, Z_R2_train: %.4f'
        % (np.sum(X_RMSE_train) / np.array(X_RMSE_train).shape, np.sum(X_R2_train) / np.array(X_R2_train).shape,
           np.sum(Y_RMSE_train) / np.array(Y_RMSE_train).shape, np.sum(Y_R2_train) / np.array(Y_R2_train).shape,
           np.sum(Z_RMSE_train) / np.array(Z_RMSE_train).shape, np.sum(Z_R2_train) / np.array(Z_R2_train).shape))
    print('test error')
    print(
        'X_RMSE_test: %.4f, X_R2_test: %.4f, Y_RMSE_test: %.4f, Y_R2_test: %.4f, Z_RMSE_test: %.4f, Z_R2_test: %.4f'
        % (np.sum(X_RMSE_test) / np.array(X_RMSE_test).shape, np.sum(X_R2_test) / np.array(X_R2_test).shape,
           np.sum(Y_RMSE_test) / np.array(Y_RMSE_test).shape, np.sum(Y_R2_test) / np.array(Y_R2_test).shape,
           np.sum(Z_RMSE_test) / np.array(Z_RMSE_test).shape, np.sum(Z_R2_test) / np.array(Z_R2_test).shape))

save_3()

plt.xlabel('Prediction', fontsize=25)
plt.ylabel('Actual', fontsize=25)
plt.savefig('%s/interpolation.tiff' %(train_save_path), dpi=300)
plt.clf()