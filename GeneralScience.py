# -*- coding: utf-8 -*-
# """
# Created on Tue Sep 13 14:11:32 2022

# @author: Samuel
# install miniconda
# conda update conda -y
# conda install -c conda-forge mamba -y
# mamba update --all -y

# conda env remove --name allen -y
# mamba create --name science --clone caiman2 -y
# conda activate science

# missing on desktop
# mamba install brian2 dipy mne psychopy scrapy seaborn statsmodels sympy -y
# pip install  nengo nengo-dl nengo-gui NeuroTools nitime pyratlib spikeinterface
# mamba list --explicit > caiman2.txt
# mamba env export > environment.yml
# mamba install pytorch torchvision torchaudio cpuonly -c pytorch
# mamba update scipy, numpy
# mmaba install tifffile==2022.4.8
# TypeError: write() got an unexpected keyword argument 'compress'
# error: OpenCV(4.6.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1267: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
# mamba install opencv==4.1


#science
# mamba install brian2 dipy GitPython keyboard matplotlib mne mplcursors numpy pandas psychopy PyGithub pytables roifile scrapy scikit-learn scikit-image scipy seaborn  spyder-kernels sqlalchemy statsmodels sympy tensorflow  -y
# mamba create --name science2 --clone science -y
# conda activate science2
#science2
# pip install  nengo nengo-dl nengo-gui networkx NeuroTools nibabel nilearn nitime pandastable pyarrow pyratlib spikeinterface tkcalendar XlsxWriter 
# mamba install caiman -y
# mamba create --name science3 --clone science2 -y
# conda activate science3
#science3
# mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
# pip install suite2p  pyro-ppl 

#allen
# mamba create -n allen python==3.7.0 ipython spyder-kernels
# pip install allensdk
# mamba install GitPython PyGithub mplcursors keyboard roifile scikit-learn sympy tensorflow scrapy mne dipy psychopy brian2 -y
# pip install pandastable tkcalendar pyarrow XlsxWriter spikeinterface NeuroTools nengo nengo-gui nengo-dl nitime nilearn nibabel pyratlib
# mamba install caiman -y
# pytables                  3.6.1 conda
# h5py                      3.7.0 pip
# mamba install h5py==2.10.0 this is the one i have worling on the lab

# pip uninstall h5py




# setx /M PATH "%PATH%;C:\Users\sp3660\AppData\Roaming\Python\Python38\Scripts
# mamba install tifffile==2024.4.8 BAD, very slow lots of siseeus 
# caiman timeseries chanche compress= fro compression=
# tifffile issues seem that caiman is working with a different mirror v0.12, but that gives environement problems, doesnt seem to be able to install with conda
# conda install tifffile==0.12 this doesn install 
# pip install tifffile==0.12
# cnat remove tifffile 2022.4.28 because it generate incosnsitencies

# CAIMAN MODEUL PATH C:\Users\sp3660\anaconda3\envs\caiman\Lib\site-packages\caiman\source_extraction\cnmf


#         to do configure gitpyhton
#     conda install -c conda-forge opencv
#     conda config --set allow_conda_downgrades true
#     conda install -n root conda=4.6
#     conda config --set auto_update_conda false
#     conda config --remove channels conda-forge
# """
    
import numpy as np
import sympy as sy
import math
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import skimage
import PIL
import statsmodels
import tensorflow
import scrapy
import seaborn
import mne
import dipy
import psychopy
import brian2
import tables 
import git 
import github 
import mplcursors 
import keyboard
import roifile

import pandastable 
import tkcalendar
import pyarrow
import xlsxwriter
import spikeinterface
import networkx
import NeuroTools
import nengo
import nitime
import nilearn
import nibabel
import pyratlib
import caiman as cm
import cv2
import suite2p
import pyro
import allensdk 
#%% skimage
from skimage import data
camera=data.camera()
type(camera)

camera.shape
camera.size
camera.min(), camera.max()
camera.mean()

camera[10,20]
camera[3,10]=0
camera[:10]=0
plt.imshow(camera)
mask=camera<87
camera[mask]=255
plt.imshow(camera)

inds_r = np.arange(len(camera))
inds_c = 6 * inds_r % len(camera)
camera[inds_r, inds_c] = 0
plt.imshow(camera)

nrows, ncols = camera.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >
                   (nrows / 2)**2)
camera[outer_disk_mask] = 0
