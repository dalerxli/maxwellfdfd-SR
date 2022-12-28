import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np

def h5_data(k):
    with h5py.File(k, 'r') as f:
        a_group_key = list(f.keys())[0]
        k = list(f[a_group_key])
    k = np.array(k)
    return k

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def error_image(error, name):
    parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
    plt.rcParams.update(parameters)
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.pcolor(error, cmap='gray')
    plt.clim(0, 0.01)
    # plt.colorbar()
    plt.axis('off')
    plt.savefig('%s/error_9_x_%s.tiff' %(folder, name), dpi=300)
    # plt.show()
    plt.close()

folder = createFolder('D:/project/SR/SRmodel/result/error_image/')
path = 'D:/project/SR/SRmodel/sample/result/'

x_SRCNN = pd.read_csv('%s/SRCNN/predict_data/sample_X.csv' %path, index_col=0)

x_FSRCNN = pd.read_csv('%s/FSRCNN/predict_data/sample_X.csv' %path, index_col=0)

x_VDSR = pd.read_csv('%s/VDSR/predict_data/sample_X.csv' %path, index_col=0)

x_LapSRN = pd.read_csv('%s/LapSRN/predict_data/sample_X.csv' %path, index_col=0)

x_FRSR = pd.read_csv('%s/FRSR_L/predict_data/sample_X.csv' %path, index_col=0)

x_interpolation = pd.read_csv('%s/interpolation/predict_data/sample_X.csv' %path, index_col=0)

x_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/x/9.h5')
# D:/project/SR/SRmodel/sample/data/800/5

x_HR = x_HR_real[220:620]

error_SRCNN = np.abs(x_HR-x_SRCNN)
error_FSRCNN = np.abs(x_HR-x_FSRCNN)
error_VDSR = np.abs(x_HR-x_VDSR)
error_LapSRN = np.abs(x_HR-x_LapSRN)
error_FRSR = np.abs(x_HR-x_FRSR)
error_interpolation = np.abs(x_HR-x_interpolation)

#%% all image
error_image(error_SRCNN,'SRCNN')
error_image(error_FSRCNN, 'FSRCNN')
error_image(error_VDSR, 'VDSR')
error_image(error_LapSRN, 'LapSRN')
error_image(error_FRSR, 'FRSR')
error_image(error_interpolation, 'interpolation')
