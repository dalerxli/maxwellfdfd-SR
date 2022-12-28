#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import scipy.interpolate
import time
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

data_location = input('server --> s or computer --> c: ')
if data_location == 's':
    print('server')
    # server
    random_image = '/home/jang/project/SRmodel/data/test_random_folder/'
    simple_image = '/home/jang/project/SRmodel/data/test_simple_folder/'
    image = [simple_image, random_image]
    save_path = createFolder('/home/jang/project/SRmodel/result/interpolation/test/')
elif data_location == 'c':
    print('computer')
    # computer
    random_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/'
    simple_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_simple_folder/'
    image = [simple_image, random_image]
    save_path = createFolder('D:/project/SR/SRmodel/result/interpolation/test_p/')
else:
    print('data location error')

xdata_file = data(10)
ydata_file = data(5)
xdata, ydata = h5_XYZ(xdata_file, ydata_file)

# (n, input shape, input shape, channels)
xdata, ydata = np.array(xdata), np.array(ydata)

xdata[xdata < 0] = 0
ydata[ydata < 0] = 0

power_RMSE = []
power_R2 = []
for i in range(xdata.shape[0]):
    power = np.sqrt(np.power(xdata[i][399:400, :, 0], 2) + np.power(xdata[i][399:400, :, 1], 2)) * xdata[i][399:400, :, 2]
    power_real = np.sqrt(np.power(ydata[i][399:400, :, 0], 2) + np.power(ydata[i][399:400, :, 1], 2)) * ydata[i][399:400, :, 2]

    power_RMSE.append(mean_squared_error(power, power_real) ** 0.5)
    power_R2.append(r2_score(power.reshape(-1, 1), power_real.reshape(-1, 1)))


pd.DataFrame(np.hstack((np.mean(power_RMSE), np.mean(power_R2))).reshape(1,-1)).to_csv('%s/power.csv' %save_path, header=['RMSE', 'R2'], index=False)