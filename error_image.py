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

folder = createFolder('D:/project/SR/SRmodel/result/error_image/')

x_SRCNN = pd.read_csv('D:/project/SR/SRmodel/result/SRCNN/test/predict_data/9_X.csv', index_col=0)
y_SRCNN = pd.read_csv('D:/project/SR/SRmodel/result/SRCNN/test/predict_data/9_Y.csv', index_col=0)
z_SRCNN = pd.read_csv('D:/project/SR/SRmodel/result/SRCNN/test/predict_data/9_Z.csv', index_col=0)

x_FSRCNN = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test/predict_data/9_X.csv', index_col=0)
y_FSRCNN = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test/predict_data/9_Y.csv', index_col=0)
z_FSRCNN = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test/predict_data/9_Z.csv', index_col=0)

x_VDSR = pd.read_csv('D:/project/SR/SRmodel/result/VDSR/test/predict_data/9_X.csv', index_col=0)
y_VDSR = pd.read_csv('D:/project/SR/SRmodel/result/VDSR/test/predict_data/9_Y.csv', index_col=0)
z_VDSR = pd.read_csv('D:/project/SR/SRmodel/result/VDSR/test/predict_data/9_Z.csv', index_col=0)

x_FRSR = pd.read_csv('D:/project/SR/SRmodel/result/FRSR_L/test/predict_data/9_X.csv', index_col=0)
y_FRSR = pd.read_csv('D:/project/SR/SRmodel/result/FRSR_L/test/predict_data/9_Y.csv', index_col=0)
z_FRSR = pd.read_csv('D:/project/SR/SRmodel/result/FRSR_L/test/predict_data/9_Z.csv', index_col=0)

x_interpolation = pd.read_csv('D:/project/SR/SRmodel/result/interpolation/test/predict_data/9_X.csv', index_col=0)
y_interpolation = pd.read_csv('D:/project/SR/SRmodel/result/interpolation/test/predict_data/9_Y.csv', index_col=0)
z_interpolation = pd.read_csv('D:/project/SR/SRmodel/result/interpolation/test/predict_data/9_Z.csv', index_col=0)

x_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/x/9.h5')
y_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/y/9.h5')
z_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/z/9.h5')

x_HR = x_HR_real[220:620]
y_HR = y_HR_real[220:620]
z_HR = z_HR_real[220:620]

error_SRCNN = np.abs(x_HR-x_SRCNN)
error_FSRCNN = np.abs(x_HR-x_FSRCNN)
error_VDSR = np.abs(x_HR-x_VDSR)
error_FRSR = np.abs(x_HR-x_FRSR)
error_interpolation = np.abs(x_HR-x_interpolation)
#%% all image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 10]
plt.pcolor(error_SRCNN, cmap='gray')
plt.clim(0, 0.01)
# plt.colorbar()
plt.axis('off')
# plt.savefig('%s/error_9_x_SRCNN.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 10]
plt.pcolor(error_FSRCNN, cmap='gray')
plt.clim(0, 0.01)
# plt.colorbar()
plt.axis('off')
# plt.savefig('%s/error_9_x_FSRCNN.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 10]
plt.pcolor(error_VDSR, cmap='gray')
plt.clim(0, 0.01)
# plt.colorbar()
plt.axis('off')
# plt.savefig('%s/error_9_x_VDSR.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 10]
plt.pcolor(error_FRSR, cmap='gray')
plt.clim(0, 0.01)
# plt.colorbar()
plt.axis('off')
# plt.savefig('%s/error_9_x_FRSR.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 10]
plt.pcolor(error_interpolation, cmap='gray')
plt.clim(0, 0.01)
# plt.colorbar()
plt.axis('off')
# plt.savefig('%s/error_9_x_interpolation.tiff' %folder, dpi=300)
plt.show()