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

folder = createFolder('D:/project/SR/SRmodel/result/real_image/')

x_SR = pd.read_csv('D:/project/SR/SRmodel/result/RFSR_L/test/predict_data/9_X.csv', index_col=0)
y_SR = pd.read_csv('D:/project/SR/SRmodel/result/RFSR_L/test/predict_data/9_Y.csv', index_col=0)
z_SR = pd.read_csv('D:/project/SR/SRmodel/result/RFSR_L/test/predict_data/9_Z.csv', index_col=0)

x_SR_FSRCNN = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test/predict_data/9_X.csv', index_col=0)
y_SR_FSRCNN = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test/predict_data/9_Y.csv', index_col=0)
z_SR_FSRCNN = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test/predict_data/9_Z.csv', index_col=0)

x_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/x/9.h5')
y_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/y/9.h5')
z_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/z/9.h5')

x_HR = x_HR_real[220:620]
y_HR = y_HR_real[220:620]
z_HR = z_HR_real[220:620]

x_LR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/10/x/9.h5')
y_LR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/10/y/9.h5')
z_LR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/10/z/9.h5')

x_LR = x_LR_real[110:310]
y_LR = y_LR_real[110:310]
z_LR = z_LR_real[110:310]

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



#%% all image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 14]
plt.pcolor(x_HR_real, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.savefig('%s/9_x_HR_real.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 14]
plt.pcolor(x_LR_real, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.savefig('%s/9_x_LR_real.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_SR, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.savefig('%s/9_X_SR.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_HR, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.savefig('%s/9_X_HR.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_LR, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.savefig('%s/9_X_LR.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_SR_FSRCNN, cmap='coolwarm')
plt.clim(0, 1.75)
plt.colorbar()
plt.savefig('%s/9_X_SR_FSRCNN.tiff' %folder, dpi=300)
plt.show()


#%% expand

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_SR.iloc[40:120, 150:230], cmap='coolwarm')
plt.clim(0, 1.75)
plt.savefig('%s/9_X_SR_expand.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_HR[40:120, 150:230], cmap='coolwarm')
plt.clim(0, 1.75)
plt.savefig('%s/9_X_HR_expand.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_LR[20:60, 75:115], cmap='coolwarm')
plt.clim(0, 1.75)
plt.savefig('%s/9_X_LR_expand.tiff' %folder, dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_SR_FSRCNN.iloc[40:120, 150:230], cmap='coolwarm')
plt.clim(0, 1.75)
plt.savefig('%s/9_X_SR_FSRCNN_expand.tiff' %folder, dpi=300)
plt.show()




#%% HR real image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 14]
plt.pcolor(x_HR_real)
plt.clim(0, 1.75)
# plt.colorbar()
# plt.savefig('%s/9_x_HR_real.tiff' %folder, dpi=300)
plt.show()

# plt.rcParams['figure.figsize'] = [10, 14]
# plt.pcolor(y_HR_real)
# plt.colorbar()
# # plt.savefig('%s/9_Y_HR_real.tiff' %folder, dpi=300)
# plt.show()
#
# plt.rcParams['figure.figsize'] = [10, 14]
# plt.pcolor(z_HR_real)
# plt.colorbar()
# # plt.savefig('%s/9_z_HR_real.tiff' %folder, dpi=300)
# plt.show()

#%% LR real image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 14]
# plt.axis([0, x_LR_real.shape[1], 0, x_LR_real.shape[0]])
plt.pcolor(x_LR_real)
plt.colorbar()
# plt.savefig('%s/9_x_LR_real.tiff' %folder, dpi=300)
plt.show()

# plt.rcParams['figure.figsize'] = [10, 14]
# plt.pcolor(y_LR_real)
# plt.colorbar()
# # plt.savefig('%s/9_Y_LR_real.tiff' %folder, dpi=300)
# plt.show()
#
# plt.rcParams['figure.figsize'] = [10, 14]
# plt.pcolor(z_LR_real)
# plt.colorbar()
# # plt.savefig('%s/9_z_LR_real.tiff' %folder, dpi=300)
# plt.show()



#%% SR image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_SR)
plt.colorbar()
# plt.savefig('%s/9_X_SR.tiff' %folder, dpi=300)
plt.show()

# plt.rcParams['figure.figsize'] = [10, 8]
# plt.pcolor(y_SR)
# plt.colorbar()
# # plt.savefig('%s/9_Y_SR.tiff' %folder, dpi=300)
# plt.show()
#
# plt.rcParams['figure.figsize'] = [10, 8]
# plt.pcolor(z_SR)
# plt.colorbar()
# # plt.savefig('%s/9_Z_SR.tiff' %folder, dpi=300)
# plt.show()

#%% HR image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_HR)
plt.colorbar()
# plt.savefig('%s/9_X_HR.tiff' %folder, dpi=300)
plt.show()

# plt.rcParams['figure.figsize'] = [10, 8]
# plt.pcolor(y_HR)
# plt.colorbar()
# # plt.savefig('%s/9_Y_HR.tiff' %folder, dpi=300)
# plt.show()
#
# plt.rcParams['figure.figsize'] = [10, 8]
# plt.pcolor(z_HR)
# plt.colorbar()
# # plt.savefig('%s/9_Z_HR.tiff' %folder, dpi=300)
# plt.show()

#%% LR image

parameters = {'xtick.labelsize': 30, 'ytick.labelsize': 30}
plt.rcParams.update(parameters)

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(x_LR)
plt.colorbar()
# plt.savefig('%s/9_X_LR.tiff' %folder, dpi=300)
plt.show()

# plt.rcParams['figure.figsize'] = [10, 8]
# plt.pcolor(y_LR)
# plt.colorbar()
# # plt.savefig('%s/9_Y_LR.tiff' %folder, dpi=300)
# plt.show()
#
# plt.rcParams['figure.figsize'] = [10, 8]
# plt.pcolor(z_LR)
# plt.colorbar()
# # plt.savefig('%s/9_Z_LR.tiff' %folder, dpi=300)
# plt.show()