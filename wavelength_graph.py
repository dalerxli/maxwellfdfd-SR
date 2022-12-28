import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def average_error(table):
    data = pd.DataFrame(np.zeros((24, 2)), columns=['RMSE', 'R2'])
    data['RMSE'] = (table['X_RMSE']+ table['Y_RMSE'] + table['Z_RMSE'])/3
    data['R2'] = (table['X_R2'] + table['Y_R2'] + table['Z_R2']) / 3
    return data

wavelength = []
for i in range(400, 1600, 50):
    wavelength.append(i)

interpoltion_table = pd.read_csv('D:/project/SR/SRmodel/result/interpolation/test_wavelength/result.csv')
srcnn_table = pd.read_csv('D:/project/SR/SRmodel/result/SRCNN/test_wavelength/result.csv')
fsrcnn_table = pd.read_csv('D:/project/SR/SRmodel/result/FSRCNN/test_wavelength/result.csv')
vdsr_table = pd.read_csv('D:/project/SR/SRmodel/result/VDSR/test_wavelength/result.csv')
LapSRN_table = pd.read_csv('D:/project/SR/SRmodel/result/LapSRN/test_wavelength/result.csv')
frsr_table = pd.read_csv('D:/project/SR/SRmodel/result/FRSR_L/test_wavelength/result.csv')

interpolation = average_error(interpoltion_table)
srcnn = average_error(srcnn_table)
fsrcnn = average_error(fsrcnn_table)
vdsr = average_error(vdsr_table)
LapSRN = average_error(LapSRN_table)
frsr = average_error(frsr_table)

# plt.plot(interpolation['RMSE'])
plt.figure(figsize=(9, 10))
plt.plot(wavelength, srcnn['RMSE'], 'k:', label = 'SRCNN', marker = 'x', linewidth = '2')
plt.plot(wavelength, fsrcnn['RMSE'], 'k-.', label = 'FSRCNN', marker = 's', linewidth = '2')
plt.plot(wavelength, vdsr['RMSE'], 'k--', label = 'VDSR', marker = 'o', linewidth = '2')
plt.plot(wavelength, LapSRN['RMSE'], 'k-', label = 'LapSRN', marker = 'o', linewidth = '2')
plt.plot(wavelength, frsr['RMSE'], 'r' ,label = 'FRSR',marker = '^', linewidth = '2.5')
plt.yticks(size = 16.5)
plt.xticks(range(400, 1600, 100), rotation = 90, size = 16.5)
plt.legend(prop={'size': 15})
plt.grid(True, alpha=0.5, linestyle='--')
plt.xlabel('Wavelength', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.savefig('D:/project/SR/SRmodel/result/wavelength_RMSE.tiff', dpi=300)
# plt.show()
plt.clf()


plt.figure(figsize=(9, 10))
plt.plot(wavelength, srcnn['R2'], 'k:', label = 'SRCNN', marker = 'x', linewidth = '2')
plt.plot(wavelength, fsrcnn['R2'], 'k-.', label = 'FSRCNN', marker = 's', linewidth = '2')
plt.plot(wavelength, vdsr['R2'], 'k--', label = 'VDSR', marker = 'o', linewidth = '2')
plt.plot(wavelength, LapSRN['R2'], 'k-', label = 'LapSRN', marker = 'o', linewidth = '2')
plt.plot(wavelength, frsr['R2'], 'r',  label = 'FRSR',marker = '^', linewidth = '2.5')
plt.yticks(size = 16.5)
plt.xticks(range(400, 1600, 100), rotation = 90, size = 16.5)
plt.legend(prop={'size': 15})
plt.grid(True, alpha=0.5, linestyle='--')
plt.xlabel('Wavelength', fontsize=25)
plt.ylabel('R2', fontsize=25)
plt.savefig('D:/project/SR/SRmodel/result/wavelength_R2.tiff', dpi=300)
plt.clf()
