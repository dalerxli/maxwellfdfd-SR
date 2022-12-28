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

def full_image(data, name):
    plt.rcParams['figure.figsize'] = [10, 14]
    plt.pcolor(data, cmap='coolwarm')
    plt.clim(0, 1.75)
    plt.colorbar()
    plt.savefig('%s/%s.tiff' % (folder, name), dpi=300)
    # plt.show()
    plt.close()

def expand_image(data, name):
    plt.rcParams['figure.figsize'] = [9, 9]
    plt.pcolor(data.iloc[40:100, 155:215], cmap='coolwarm')
    plt.clim(0, 1.75)
    plt.xticks([0, 15, 30, 45, 60], labels=[155, 170, 185, 200, 215])
    plt.yticks([0, 15, 30, 45, 60], labels=[40, 55, 70, 85, 100])
    plt.tick_params(length=20)
    plt.savefig('%s/%s.tiff' % (folder, name), dpi=300)
    # plt.show()
    plt.close()

def layout_image(data, name):
    plt.rcParams['figure.figsize'] = [9, 9]
    plt.pcolor(data.iloc[40:100, 155:215], cmap='coolwarm')
    plt.clim(0, 1.75)
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('%s/layout/%s.tiff' % (folder, name), dpi=300)
    # plt.show()
    plt.close()

path = 'D:/project/SR/SRmodel/sample/result/'
folder = createFolder('%s/sample_image/' %path)
createFolder('%s/layout/' %folder)

x_SRCNN = pd.read_csv('%s/SRCNN/predict_data/sample_X.csv' %path, index_col=0)

x_FSRCNN = pd.read_csv('%s/FSRCNN/predict_data/sample_X.csv' %path, index_col=0)

x_VDSR = pd.read_csv('%s/VDSR/predict_data/sample_X.csv' %path, index_col=0)

x_LapSRN = pd.read_csv('%s/LapSRN/predict_data/sample_X.csv' %path, index_col=0)

x_FRSR = pd.read_csv('%s/FRSR_L/predict_data/sample_X.csv' %path, index_col=0)

x_interpolation = pd.read_csv('%s/interpolation/predict_data/sample_X.csv' %path, index_col=0)

x_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/x/9.h5')

x_HR = x_HR_real[220:620]

x_LR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/10/x/9.h5')

x_LR = x_LR_real[110:310]

#%% base image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 14]
plt.pcolor(x_HR_real, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.xticks([0, 100, 200, 300, 400], labels=['', -500, 0, 500, ''])
plt.yticks([0, 100, 200, 300, 400, 500, 600, 700], labels=['', -500, 0, 500, 1000, 1500, 2000, 2500])
plt.axhline(y=100, color='black', linewidth=2)
plt.savefig('%s/base_image.tiff' %folder, dpi=300)
plt.show()
plt.close()


#%% all image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

full_image(x_HR_real, '9_x_HR_real')
full_image(x_LR_real, '9_x_LR_real')
full_image(x_SRCNN, '9_X_SRCNN')
full_image(x_FSRCNN, '9_X_FSRCNN')
full_image(x_VDSR, '9_X_VDSR')
full_image(x_LapSRN, '9_X_LapSRN')
full_image(x_FRSR, '9_X_FRSR')
full_image(x_HR, '9_X_HR')
full_image(x_LR, '9_X_LR')
full_image(x_interpolation, '9_X_interpolation')


#%% expand

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

expand_image(x_SRCNN, '9_X_SRCNN_expand')
expand_image(x_FSRCNN, '9_X_FSRCNN_expand')
expand_image(x_VDSR, '9_X_VDSR_expand')
expand_image(x_LapSRN, '9_X_LapSRN_expand')
expand_image(x_FRSR, '9_X_FRSR_expand')
expand_image(x_interpolation, '9_X_interpolation_expand')

plt.rcParams['figure.figsize'] = [9, 9]
plt.pcolor(x_HR[40:100, 155:215], cmap='coolwarm')
plt.clim(0, 1.75)
plt.xticks([0, 15, 30, 45, 60], labels=[155, 170, 185, 200, 215])
plt.yticks([0, 15, 30, 45, 60], labels=[40, 55, 70, 85, 100])
plt.tick_params(length=20)
plt.savefig('%s/9_X_HR_expand.tiff' %folder, dpi=300)
# plt.show()
plt.close()

plt.rcParams['figure.figsize'] = [9, 9]
plt.pcolor(x_LR[20:50, 77:107], cmap='coolwarm')
plt.clim(0, 1.75)
plt.xticks([0, 7.5, 15, 22.5, 30], labels=[155, 170, 185, 200, 215])
plt.yticks([0, 7.5, 15, 22.5, 30], labels=[40, 55, 70, 85, 100])
plt.tick_params(length=20)
plt.savefig('%s/9_X_LR_expand.tiff' %folder, dpi=300)
# plt.show()
plt.close()


# no layout

layout_image(x_SRCNN, '9_X_SRCNN_expand')
layout_image(x_FSRCNN, '9_X_FSRCNN_expand')
layout_image(x_VDSR, '9_X_VDSR_expand')
layout_image(x_LapSRN, '9_X_LapSRN_expand')
layout_image(x_FRSR, '9_X_FRSR_expand')
layout_image(x_interpolation, '9_X_interpolation_expand')

plt.rcParams['figure.figsize'] = [9, 9]
plt.pcolor(x_LR[20:50, 77:107], cmap='coolwarm')
plt.clim(0, 1.75)
plt.axis('off'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
plt.savefig('%s/layout/9_X_LR_expand.tiff' %folder, dpi=300)
# plt.show()
plt.close()

plt.rcParams['figure.figsize'] = [9, 9]
plt.pcolor(x_HR[40:100, 155:215], cmap='coolwarm')
plt.clim(0, 1.75)
plt.axis('off'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
plt.savefig('%s/layout/9_X_HR_expand.tiff' %folder, dpi=300)
# plt.show()
plt.close()
