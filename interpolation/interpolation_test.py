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

def PSNR(compressed, original):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel_ori = np.max(original)
    max_pixel_com = np.max(compressed)
    psnr_ori = 20 * np.log10(max_pixel_ori / np.sqrt(mse))
    psnr_com = 20 * np.log10(max_pixel_com / np.sqrt(mse))
    return psnr_com, psnr_ori

def evaluation(k):
    RMSE = mean_squared_error(xdata[j][:, :, k], ydata[j][:, :, k]) ** 0.5
    R2 = r2_score(xdata[j][:, :, k], ydata[j][:, :, k])
    psnr = PSNR(xdata[j][:, :, k], ydata[j][:, :, k])[0]

    return RMSE, R2, psnr

def evaluation_all(RMSE, R2, psnr):
    RMSE_ = np.sum(RMSE) / np.array(RMSE).shape
    R2_ = np.sum(R2) / np.array(R2).shape
    psnr_ = np.sum(psnr) / np.array(psnr).shape

    return RMSE_, R2_, psnr_

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

X_RMSE_ydata, X_R2_ydata, X_psnr = [], [], []
Y_RMSE_ydata, Y_R2_ydata, Y_psnr = [], [], []
Z_RMSE_ydata, Z_R2_ydata, Z_psnr = [], [], []

parameters = {'xtick.labelsize': 20, 'ytick.labelsize': 20}
plt.rcParams.update(parameters)
plt.figure(figsize=(8, 8))

for j in range(np.array(ydata).shape[0]):
    X_RMSE_ydata.append(evaluation(0)[0])
    X_R2_ydata.append(evaluation(0)[1])
    X_psnr.append(evaluation(0)[2])

    Y_RMSE_ydata.append(evaluation(1)[0])
    Y_R2_ydata.append(evaluation(1)[1])
    Y_psnr.append(evaluation(1)[2])

    Z_RMSE_ydata.append(evaluation(2)[0])
    Z_R2_ydata.append(evaluation(2)[1])
    Z_psnr.append(evaluation(2)[2])

    if divmod(j, 100)[1] == 0:
        plt.scatter(xdata[j][:, :, 0].reshape(1, -1), ydata[j][:, :, 0].reshape(1, -1), c='black', s=1, alpha=0.4)
        plt.scatter(xdata[j][:, :, 1].reshape(1, -1), ydata[j][:, :, 1].reshape(1, -1), c='black', s=1, alpha=0.4)
        plt.scatter(xdata[j][:, :, 2].reshape(1, -1), ydata[j][:, :, 2].reshape(1, -1), c='black', s=1, alpha=0.4)

X_RMSE_ydata_all, X_R2_ydata_all, X_psnr_all = evaluation_all(X_RMSE_ydata, X_R2_ydata, X_psnr)
Y_RMSE_ydata_all, Y_R2_ydata_all, Y_psnr_all = evaluation_all(Y_RMSE_ydata, Y_R2_ydata, Y_psnr)
Z_RMSE_ydata_all, Z_R2_ydata_all, Z_psnr_all = evaluation_all(Z_RMSE_ydata, Z_R2_ydata, Z_psnr)

print('ydata error')
print(
    'X_RMSE_ydata: %.4f, X_R2_ydata: %.4f, X_psnr: %.4f, '
    'Y_RMSE_ydata: %.4f, Y_R2_ydata: %.4f, Y_psnr: %.4f, '
    'Z_RMSE_ydata: %.4f, Z_R2_ydata,: %.4f, Z_psnr: %.4f '

    % (X_RMSE_ydata_all, X_R2_ydata_all, X_psnr_all,
       Y_RMSE_ydata_all, Y_R2_ydata_all, Y_psnr_all,
       Z_RMSE_ydata_all, Z_R2_ydata_all, Z_psnr_all))

for i in [1, -1]:
    prediction_save_path = createFolder('%s/predict_data/' % (save_path))
    pd.DataFrame(xdata[i][:, :, 0]).to_csv('%s/%s_X.csv' % (prediction_save_path, i + 1))
    pd.DataFrame(xdata[i][:, :, 1]).to_csv('%s/%s_Y.csv' % (prediction_save_path, i + 1))
    pd.DataFrame(xdata[i][:, :, 2]).to_csv('%s/%s_Z.csv' % (prediction_save_path, i + 1))

pd.DataFrame(xdata[800][:, :, 0]).to_csv('%s/9_X.csv' % (prediction_save_path))
pd.DataFrame(xdata[800][:, :, 1]).to_csv('%s/9_Y.csv' % (prediction_save_path))
pd.DataFrame(xdata[800][:, :, 2]).to_csv('%s/9_Z.csv' % (prediction_save_path))


plt.xlabel('Prediction', fontsize=30)
plt.ylabel('Actual', fontsize=30)
plt.xlim([0, 3])
plt.ylim([0, 3])
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], labels=[0, '', 1, '', 2, '', 3])
plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], labels=[0, '', 1, '', 2, '', 3])
plt.savefig('%s/interpolation.tiff' % (save_path), dpi=300)
plt.clf()

data_all = [[float(X_RMSE_ydata_all), float(X_R2_ydata_all), float(X_psnr_all),
            float(Y_RMSE_ydata_all), float(Y_R2_ydata_all), float(Y_psnr_all),
            float(Z_RMSE_ydata_all), float( Z_R2_ydata_all), float(Z_psnr_all)]]

pd.DataFrame(data_all).to_csv('%s/result.csv' % (save_path),
                                  header=['X_RMSE', 'X_R2', 'X_psnr',
                                          'Y_RMSE', 'Y_R2', 'Y_psnr',
                                          'Z_RMSE', 'Z_R2', 'Z_psnr'], index=False)

# ydata error
# X_RMSE_ydata: 0.0536, X_R2_ydata: 0.5151, Y_RMSE_ydata: 0.0120, Y_R2_ydata: 0.9011, Z_RMSE_ydata: 0.0207, Z_R2_ydata: 0.7850