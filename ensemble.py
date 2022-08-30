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

model_name = input('SRCNN: S , VDSR: V , FSRCNN: F , FRSR: R :')

if model_name == 'S' or model_name == 'SRCNN' or model_name == 'srcnn' or model_name == 's':
    print('SRCNN')
    model_name = 'SRCNN'
elif model_name == 'V' or model_name == 'VDSR' or model_name == 'vdsr' or model_name == 'v':
    print('VDSR')
    model_name = 'VDSR'
elif model_name == 'F' or model_name == 'FSRCNN' or model_name == 'fsrcnn' or model_name == 'f':
    print('FSRCNN')
    model_name = 'FSRCNN'
elif model_name == 'R' or model_name == 'FRSR' or model_name == 'frsr' or model_name == 'r':
    print('FRSR')
    model_name = 'FRSR'
else:
    print('model_name error')

data_location = input('server --> s or computer --> c: ')
if data_location == 's':
    print('server')
    # server
    random_image = '/home/jang/project/SRmodel/data/test_random_folder/'
    simple_image = '/home/jang/project/SRmodel/data/test_simple_folder/'
    image = [simple_image, random_image]
    dir_path = "/home/jang/project/SRmodel/result/%s/save_model/" % (model_name)
    save_path = createFolder('/home/jang/project/SRmodel/result/%s/test/' % (model_name))
elif data_location == 'c':
    print('computer')
    # computer
    random_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/'
    simple_image = 'D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_simple_folder/'
    image = [simple_image, random_image]
    dir_path = 'D:/project/SR/SRmodel/result/%s/save_model/' % (model_name)
    save_path = createFolder('D:/project/SR/SRmodel/result/%s/test/' % (model_name))
else:
    print('data location error')

model_path = []
for (root, directories, files) in os.walk('%s'%(dir_path)):
    # for d in directories:
    #     d_path = os.path.join(root, d)
    #     print(d_path)

    for file in files:
        if '.h5' in file:
            file_path = os.path.join(root, file)
            model_path.append(file_path)
xdata_file = data(10)
ydata_file = data(5)
xdata, ydata = h5_XYZ(xdata_file, ydata_file)

# (n, input shape, input shape, channels)
xdata, ydata = np.array(xdata), np.array(ydata)

scalers_x = StandardScaler()
scalers_y = StandardScaler()
scalers_z = StandardScaler()

scalers_x.fit(xdata[:, :, :, 0].reshape(-1, 1))
scalers_y.fit(xdata[:, :, :, 1].reshape(-1, 1))
scalers_z.fit(xdata[:, :, :, 2].reshape(-1, 1))

xdata[:, :, :, 0:1] = scalers_x.transform(xdata[:, :, :, 0].reshape(-1, 1)).reshape(xdata.shape[0],
                                                                                    xdata.shape[1],
                                                                                    xdata.shape[2], 1)
xdata[:, :, :, 1:2] = scalers_y.transform(xdata[:, :, :, 1].reshape(-1, 1)).reshape(xdata.shape[0],
                                                                                    xdata.shape[1],
                                                                                    xdata.shape[2], 1)
xdata[:, :, :, 2:3] = scalers_z.transform(xdata[:, :, :, 2].reshape(-1, 1)).reshape(xdata.shape[0],
                                                                                    xdata.shape[1],
                                                                                    xdata.shape[2], 1)

ydata_result_all = {}
start_time = time.time()
for i in range(len(model_path)):
    print(i+1)
    model =load_model(filepath= model_path[i])
    result = model.predict(xdata, verbose=0)
    result[:, :, :, 0:1] = scalers_x.inverse_transform(result[:, :, :, 0].reshape(-1, 1)).reshape(result.shape[0],
                                                                                                  result.shape[1],
                                                                                                  result.shape[2], 1)
    result[:, :, :, 1:2] = scalers_y.inverse_transform(result[:, :, :, 1].reshape(-1, 1)).reshape(result.shape[0],
                                                                                                  result.shape[1],
                                                                                                  result.shape[2], 1)
    result[:, :, :, 2:3] = scalers_z.inverse_transform(result[:, :, :, 2].reshape(-1, 1)).reshape(result.shape[0],
                                                                                                  result.shape[1],
                                                                                                  result.shape[2], 1)
    ydata_result_all[i] = np.array(result)/len(model_path)

ydata_result = np.zeros(np.shape(ydata_result_all[0]))
for i in range(len(ydata_result_all)):
    ydata_result = ydata_result_all[i] + ydata_result

ydata_result[ydata_result < 0] = 0

X_RMSE_ydata, X_R2_ydata = [], []
Y_RMSE_ydata, Y_R2_ydata = [], []
Z_RMSE_ydata, Z_R2_ydata = [], []

parameters = {'xtick.labelsize': 20, 'ytick.labelsize': 20}
plt.rcParams.update(parameters)
plt.figure(figsize=(8, 8))

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
    if divmod(j, 100)[1] == 0:
        plt.scatter(ydata_result[j][:, :, 0].reshape(1, -1), ydata[j][:, :, 0].reshape(1, -1), c='black', s=1, alpha=0.4)
        plt.scatter(ydata_result[j][:, :, 1].reshape(1, -1), ydata[j][:, :, 1].reshape(1, -1), c='black', s=1, alpha=0.4)
        plt.scatter(ydata_result[j][:, :, 2].reshape(1, -1), ydata[j][:, :, 2].reshape(1, -1), c='black', s=1, alpha=0.4)

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

print("--- %s seconds ---" % (time.time() - start_time))
model_time = time.time() - start_time
data_all = [[float(X_RMSE_ydata_all), float(X_R2_ydata_all),
            float(Y_RMSE_ydata_all), float(Y_R2_ydata_all),
            float(Z_RMSE_ydata_all),float( Z_R2_ydata_all), model_time]]
pd.DataFrame(data_all).to_csv('%s/result.csv' % (save_path),
                                  header=['X_RMSE', 'X_R2',
                                          'Y_RMSE', 'Y_R2',
                                          'Z_RMSE', 'Z_R2', 'time'], index=False)

prediction_save_path = createFolder('%s/predict_data/' % (save_path))

for i in [1, -1]:
    pd.DataFrame(ydata_result[i][:, :, 0]).to_csv('%s/%s_X.csv' % (prediction_save_path, i + 1))
    pd.DataFrame(ydata_result[i][:, :, 1]).to_csv('%s/%s_Y.csv' % (prediction_save_path, i + 1))
    pd.DataFrame(ydata_result[i][:, :, 2]).to_csv('%s/%s_Z.csv' % (prediction_save_path, i + 1))

pd.DataFrame(ydata_result[800][:, :, 0]).to_csv('%s/9_X.csv' % (prediction_save_path))
pd.DataFrame(ydata_result[800][:, :, 1]).to_csv('%s/9_Y.csv' % (prediction_save_path))
pd.DataFrame(ydata_result[800][:, :, 2]).to_csv('%s/9_Z.csv' % (prediction_save_path))

plt.xlabel('Prediction', fontsize=30)
plt.ylabel('Actual', fontsize=30)
plt.xlim([0, 3])
plt.ylim([0, 3])
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], labels=[0, '', 1, '', 2, '', 3])
plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], labels=[0, '', 1, '', 2, '', 3])
plt.savefig('%s/%s.tiff' % (save_path, model_name), dpi=300)
# plt.show()
plt.clf()
