import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import cv2

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

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


def PSNR_csv(compressed, original):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel_com = np.max(compressed)
    max_pixel_ori = np.max(original)
    psnr = 20 * np.log10(np.maximum(max_pixel_com, max_pixel_ori) / np.sqrt(mse))
    return psnr

def PSNR(compressed, original):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# with csv data
# csv_path = 'D:/project/SR/SRmodel/sample/result/'
csv_path = '/home/jang/project/SRmodel/sample/result/'
csv_SRCNN = pd.read_csv('%s/SRCNN/predict_data/sample_X.csv' %csv_path, index_col=0)
csv_FSRCNN = pd.read_csv('%s/FSRCNN/predict_data/sample_X.csv' %csv_path, index_col=0)
csv_VDSR = pd.read_csv('%s/VDSR/predict_data/sample_X.csv' %csv_path, index_col=0)
csv_LapSRN = pd.read_csv('%s/LapSRN/predict_data/sample_X.csv' %csv_path, index_col=0)
csv_FRSR = pd.read_csv('%s/FRSR_L/predict_data/sample_X.csv' %csv_path, index_col=0)
csv_interpolation = pd.read_csv('%s/interpolation/predict_data/sample_X.csv' %csv_path, index_col=0)
# csv_HR_real = h5_data('D:/project/SR/cdal_maxwellfdfd/maxwellfdfd/example/2d/test_random_folder/0001/800/5/x/9.h5')
csv_HR_real = h5_data('/home/jang/project/SRmodel/data/test_random_folder/0001/800/5/x/9.h5')

csv_HR = csv_HR_real[220:620]

psnr_SRCNN = PSNR_csv(np.array(csv_SRCNN.iloc[40:100, 155:215]), np.array(csv_HR[40:100, 155:215]))
psnr_FSRCNN = PSNR_csv(np.array(csv_FSRCNN.iloc[40:100, 155:215]), np.array(csv_HR[40:100, 155:215]))
psnr_VDSR = PSNR_csv(np.array(csv_VDSR.iloc[40:100, 155:215]), np.array(csv_HR[40:100, 155:215]))
psnr_LapSRN= PSNR_csv(np.array(csv_LapSRN.iloc[40:100, 155:215]), np.array(csv_HR[40:100, 155:215]))
psnr_FRSR = PSNR_csv(np.array(csv_FRSR.iloc[40:100, 155:215]), np.array(csv_HR[40:100, 155:215]))
psnr_interpolation = PSNR_csv(np.array(csv_interpolation.iloc[40:100, 155:215]), np.array(csv_HR[40:100, 155:215]))

# with expand image data

# sample_path = 'D:/project/SR/SRmodel/sample/result/sample_image/'
sample_path = '/home/jang/project/SRmodel/sample/result/sample_image/'
SRCNN = cv2.imread('%s/9_X_SRCNN_expand.tiff' %sample_path)
FSRCNN = cv2.imread('%s/9_X_FSRCNN_expand.tiff' %sample_path)
VDSR = cv2.imread('%s/9_X_VDSR_expand.tiff' %sample_path)
LapSRN = cv2.imread('%s/9_X_LapSRN_expand.tiff' %sample_path)
FRSR = cv2.imread('%s/9_X_FRSR_expand.tiff' %sample_path)
interpolation = cv2.imread('%s/9_X_interpolation_expand.tiff' %sample_path)
LR = cv2.imread('%s/9_X_LR_expand.tiff' %sample_path)
HR = cv2.imread('%s/9_X_HR_expand.tiff' %sample_path)

# with image data

# sample_path = 'D:/project/SR/SRmodel/sample/result/sample_image/layout/'
sample_path = '/home/jang/project/SRmodel/sample/result/sample_image/layout/'
SRCNN = cv2.imread('%s/9_X_SRCNN_expand.tiff' %sample_path)
FSRCNN = cv2.imread('%s/9_X_FSRCNN_expand.tiff' %sample_path)
VDSR = cv2.imread('%s/9_X_VDSR_expand.tiff' %sample_path)
LapSRN = cv2.imread('%s/9_X_LapSRN_expand.tiff' %sample_path)
FRSR = cv2.imread('%s/9_X_FRSR_expand.tiff' %sample_path)
interpolation = cv2.imread('%s/9_X_interpolation_expand.tiff' %sample_path)
LR = cv2.imread('%s/9_X_LR_expand.tiff' %sample_path)
HR = cv2.imread('%s/9_X_HR_expand.tiff' %sample_path)

# csv data PSNR
print('csv data PSNR \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n'
      %(psnr_SRCNN, psnr_FSRCNN, psnr_VDSR, psnr_LapSRN, psnr_FRSR, psnr_interpolation))

evaluation_result = np.vstack(('PSNR', np.hstack((psnr(SRCNN, HR), psnr(FSRCNN, HR), psnr(VDSR, HR), psnr(LapSRN, HR), psnr(FRSR, HR), psnr(interpolation, HR), psnr(LR, HR))),
                               'SSIM', np.hstack((ssim(SRCNN, HR), ssim(FSRCNN, HR), ssim(VDSR, HR), ssim(LapSRN, HR), ssim(FRSR, HR), ssim(interpolation, HR), ssim(LR, HR))),
                               'VIF', np.hstack((vifp(SRCNN, HR), vifp(FSRCNN, HR), vifp(VDSR, HR), vifp(LapSRN, HR), vifp(FRSR, HR), vifp(interpolation, HR), vifp(LR, HR))),
                               'UQI', np.hstack((uqi(SRCNN, HR), uqi(FSRCNN, HR), uqi(VDSR, HR), uqi(LapSRN, HR), uqi(FRSR, HR), uqi(interpolation, HR), uqi(LR, HR))),
                               'RASE', np.hstack((rase(SRCNN, HR), rase(FSRCNN, HR), rase(VDSR, HR), rase(LapSRN, HR), rase(FRSR, HR), rase(interpolation, HR), rase(LR, HR))),
                               'SAM', np.hstack((sam(SRCNN, HR), sam(FSRCNN, HR), sam(VDSR, HR), sam(LapSRN, HR), sam(FRSR, HR), sam(interpolation, HR), sam(LR, HR))),
                               'SCC', np.hstack((scc(SRCNN, HR), scc(FSRCNN, HR), scc(VDSR, HR), scc(LapSRN, HR), scc(FRSR, HR), scc(interpolation, HR), scc(LR, HR)))
                               ))

pd.DataFrame(evaluation_result).to_csv('%s/evaluation_result.csv' %csv_path, header=['evaluation', 'SRCNN', 'FSRCNN', 'VDSR', 'LapSRN', 'FRSR', 'interpolation', 'LR'], index=False)

# Image data PSNR
print('Image data PSNR \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(PSNR(SRCNN, HR), PSNR(FSRCNN, HR), PSNR(VDSR, HR), PSNR(LapSRN, HR), PSNR(FRSR, HR), PSNR(interpolation, HR), PSNR(LR, HR)))

# Image data SSIM
print('Image data SSIM \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(calculate_ssim(SRCNN, HR), calculate_ssim(FSRCNN, HR), calculate_ssim(VDSR, HR), calculate_ssim(LapSRN, HR), calculate_ssim(FRSR, HR), calculate_ssim(interpolation, HR), calculate_ssim(LR, HR)))

# sewar evaluation
print('sewar evaluation \n')
# PSNR(Peak Signal-to-Noise Ratio)
print('Image data PSNR \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(psnr(SRCNN, HR), psnr(FSRCNN, HR), psnr(VDSR, HR), psnr(LapSRN, HR), psnr(FRSR, HR), psnr(interpolation, HR), psnr(LR, HR)))

# SSIM(Structural Similarity Index)
print('Image data SSIM \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(ssim(SRCNN, HR), ssim(FSRCNN, HR), ssim(VDSR, HR), ssim(LapSRN, HR), ssim(FRSR, HR), ssim(interpolation, HR), ssim(LR, HR)))

# VIF(visual information fidelity)
print('Image data VIF \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(vifp(SRCNN, HR), vifp(FSRCNN, HR), vifp(VDSR, HR), vifp(LapSRN, HR), vifp(FRSR, HR), vifp(interpolation, HR), vifp(LR, HR)))

# UQI(Universal Quality Image Index)
print('Image data UQI \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(uqi(SRCNN, HR), uqi(FSRCNN, HR), uqi(VDSR, HR), uqi(LapSRN, HR), uqi(FRSR, HR), uqi(interpolation, HR), uqi(LR, HR)))

# RASE(Relative Average Spectral Error)
print('Image data RASE \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(rase(SRCNN, HR), rase(FSRCNN, HR), rase(VDSR, HR), rase(LapSRN, HR), rase(FRSR, HR), rase(interpolation, HR), rase(LR, HR)))

# SAM(Spectral Angle Mapper)
print('Image data SAM \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(sam(SRCNN, HR), sam(FSRCNN, HR), sam(VDSR, HR), sam(LapSRN, HR), sam(FRSR, HR), sam(interpolation, HR), sam(LR, HR)))

# SCC(Spatial Correlation Coefficient)
print('Image data SCC \n SRCNN: %s \n FSRCNN: %s \n VDSR: %s \n LapSRN: %s \n FRSR: %s \n interpolation: %s \n LR: %s \n'
      %(scc(SRCNN, HR), scc(FSRCNN, HR), scc(VDSR, HR), scc(LapSRN, HR), scc(FRSR, HR), scc(interpolation, HR), scc(LR, HR)))

