#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

def interpolation_matrix(X, Y):
    Source = X
    x = np.arange(0, Source.shape[0])
    fit_x = scipy.interpolate.interp1d(x, Source, axis=0, kind='cubic')
    Target = fit_x(np.linspace(0, Source.shape[0] - 1, Y.shape[0]))

    y = np.arange(0, Target.shape[1])
    fit_y = scipy.interpolate.interp1d(y, Target, axis=1, kind='cubic')
    Target = fit_y(np.linspace(0, Target.shape[1] - 1, Y.shape[1]))

    return Target

def data(cs_value, path):
    x_path = []
    for j in range(len(wavelength)):
        path_data = '%s/data/%s/%s/x/' %(path, wavelength[j], cs_value)
        for (root, directories, files) in os.walk(path_data):
            for file in files:
                if '.h5' in file:
                    file_path = os.path.join(root, file)
                    x_path.append(file_path)

    y_path = []
    for j in range(len(wavelength)):
        path_data = '%s/data/%s/%s/y/' % (path, wavelength[j], cs_value)
        for (root, directories, files) in os.walk(path_data):
            for file in files:
                if '.h5' in file:
                    file_path = os.path.join(root, file)
                    y_path.append(file_path)

    z_path = []
    for j in range(len(wavelength)):
        path_data = '%s/data/%s/%s/z/' % (path, wavelength[j], cs_value)
        for (root, directories, files) in os.walk(path_data):
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

def h5_XYZ(train_matrix, test_matrix):
    test = []
    for i in range(len(test_matrix[0])):
        test.append(np.stack([h5_data(test_matrix[0][i])[220:620],
                              h5_data(test_matrix[1][i])[220:620],
                              h5_data(test_matrix[2][i])[220:620]], axis=2))
    if model_name == 'SRCNN' or model_name == 'VDSR':
        train = []
        for i in range(len(test_matrix[0])):
            train.append(np.stack([interpolation_matrix(h5_data(train_matrix[0][i])[110:310], h5_data(test_matrix[0][i])[220:620]),
                                   interpolation_matrix(h5_data(train_matrix[1][i])[110:310], h5_data(test_matrix[1][i])[220:620]),
                                   interpolation_matrix(h5_data(train_matrix[2][i])[110:310], h5_data(test_matrix[2][i])[220:620])],
                         axis=2))
        ori_train = []
        for i in range(len(test_matrix[0])):
            ori_train.append(np.stack([h5_data(train_matrix[0][i])[110:310],
                                   h5_data(train_matrix[1][i])[110:310],
                                   h5_data(train_matrix[2][i])[110:310]], axis=2))
    else:
        train = []
        for i in range(len(test_matrix[0])):
            train.append(np.stack([h5_data(train_matrix[0][i])[110:310],
                                   h5_data(train_matrix[1][i])[110:310],
                                   h5_data(train_matrix[2][i])[110:310]], axis=2))
        ori_train = []
        for i in range(len(test_matrix[0])):
            ori_train.append(np.stack([h5_data(train_matrix[0][i])[110:310],
                                       h5_data(train_matrix[1][i])[110:310],
                                       h5_data(train_matrix[2][i])[110:310]], axis=2))
    return train, test, ori_train

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

#%%
wavelength = [800,1000]
cs = [10, 5]
EH = ['x', 'y', 'z']
model_name = input('SRCNN: S , VDSR: V , FSRCNN: F : ')

if model_name == 'S' or model_name == 'SRCNN' or model_name == 'srcnn' or model_name == 's':
    print('SRCNN')
    model_name = 'SRCNN'
elif model_name == 'V' or model_name == 'VDSR' or model_name == 'vdsr' or model_name == 'v':
    print('VDSR')
    model_name = 'VDSR'
elif model_name == 'F' or model_name == 'FSRCNN' or model_name == 'fsrcnn' or model_name == 'f':
    print('FSRCNN')
    model_name = 'FSRCNN'
else:
    print('model_name error')

data_location = input('server --> s or computer --> c: ')
if data_location == 's':
    print('server')
    # server
    path = '/home/jang/project/SRmodel/result/real_image/'
    dir_path = "/home/jang/project/SRmodel/result/%s/save_model/" % (model_name)
    save_path = createFolder('%s/%s/' % (path, model_name))
elif data_location == 'c':
    print('computer')
    # computer
    path = 'D:/project/SR/SRmodel/result/real_image/'
    dir_path = 'D:/project/SR/SRmodel/result/%s/save_model/' % (model_name)
    save_path = createFolder('%s/%s/' % (path, model_name))
else:
    print('data location error')

# prediction_save_path = createFolder('%s/predict_data/' % (save_path))

#%%
model_path = []
for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.h5' in file:
            file_path = os.path.join(root, file)
            model_path.append(file_path)

xdata_file = data(10, path)
ydata_file = data(5, path)
xdata, ydata, x_data_b = h5_XYZ(xdata_file, ydata_file)

# (n, input shape, input shape, channels)
xdata, ydata = np.array(xdata), np.array(ydata)

ydata_result = []
X_RMSE_ydata, X_R2_ydata = [], []
Y_RMSE_ydata, Y_R2_ydata = [], []
Z_RMSE_ydata, Z_R2_ydata = [], []


model =load_model(filepath= model_path[0])
ydata_result = model.predict(xdata, verbose=0)
for j in range(np.array(ydata).shape[0]):
    X_RMSE_ydata.append(mean_squared_error(ydata_result[j][:, :, 0],
                                          ydata[j][:, :, 0]) ** 0.5)
    X_R2_ydata.append(r2_score(ydata_result[j][:, :, 0],
                              ydata[j][:, :, 0]))
    Y_RMSE_ydata.append(mean_squared_error(ydata_result[j][:, :, 1],
                                          ydata[j][:, :, 1]) ** 0.5)
    Y_R2_ydata.append(r2_score(ydata_result[j][:, :, 1],
                              ydata[j][:, :, 1]))
    Z_RMSE_ydata.append(mean_squared_error(ydata_result[j][:, :, 2],
                                          ydata[j][:, :, 2]) ** 0.5)
    Z_R2_ydata.append(r2_score(ydata_result[j][:, :, 2],
                              ydata[j][:, :, 2]))

        # pd.DataFrame(ydata[j][:, :, 0]).to_csv('%s/%s_X.csv' % (prediction_save_path, j + 1))
        # pd.DataFrame(ydata[j][:, :, 1]).to_csv('%s/%s_Y.csv' % (prediction_save_path, j + 1))
        # pd.DataFrame(ydata[j][:, :, 2]).to_csv('%s/%s_Z.csv' % (prediction_save_path, j + 1))

X_RMSE_ydata_all = np.sum(X_RMSE_ydata) / np.array(X_RMSE_ydata).shape
X_R2_ydata_all = np.sum(X_R2_ydata) / np.array(X_R2_ydata).shape

Y_RMSE_ydata_all = np.sum(Y_RMSE_ydata) / np.array(Y_RMSE_ydata).shape
Y_R2_ydata_all = np.sum(Y_R2_ydata) / np.array(Y_R2_ydata).shape

Z_RMSE_ydata_all = np.sum(Z_RMSE_ydata) / np.array(Z_RMSE_ydata).shape
Z_R2_ydata_all = np.sum(Z_R2_ydata) / np.array(Z_R2_ydata).shape

print('ydata error')
print(
    'X_RMSE_ydata: %.4f, X_R2_ydata: %.4f, Y_RMSE_ydata: %.4f, Y_R2_ydata: %.4f, Z_RMSE_ydata: %.4f, Z_R2_ydata: %.4f'
    % (X_RMSE_ydata_all, X_R2_ydata_all, Y_RMSE_ydata_all, Y_R2_ydata_all, Z_RMSE_ydata_all, Z_R2_ydata_all))



#%%
for axis_xyz in EH:
    if axis_xyz == 'z':
        fig_name = 'Hz'
    else:
        fig_name = 'E%s' %(axis_xyz)

    # parameters = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.titlesize': 25}
    parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
    plt.rcParams.update(parameters)

    HR = ydata[0][:, :, EH.index(axis_xyz)]
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.pcolor(HR)
    plt.colorbar()
    # plt.title('%s_HR' % (fig_name))
    plt.savefig('%s/%s_HR.tiff' % (save_path, fig_name), dpi=300)
    plt.clf()

    prediction_y = ydata_result[0][:, :, EH.index(axis_xyz)]
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.pcolor(prediction_y)
    plt.colorbar()
    # plt.title('%s_prediction' % (fig_name))
    plt.savefig('%s/%s_prediction.tiff' % (save_path, fig_name), dpi=300)
    plt.clf()

    LR = x_data_b[0][:, :, EH.index(axis_xyz)]
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.pcolor(LR)
    plt.colorbar()
    # plt.title('%s_LR' % (fig_name))
    plt.savefig('%s/%s_LR.tiff' % (save_path, fig_name), dpi=300)
    plt.clf()
