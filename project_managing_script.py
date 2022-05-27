# -*- coding: utf-8 -*-
# """
# Created on Fri Apr  9 09:27:59 2021

# @author: sp3660
# requirements
# check environemnet and anconda
# install miniconda

# check gtihub tokeny
# git
# numpy
# matplotlib


# conda update conda
# conda update --all
# conda config --set channel_priority flexible
# conda install mamba
# mamba create -n caiman2 -c conda-forge caiman
# mamba install -c conda-forge caiman
# mamba activate caiman2
# mamba install Spyder GitPython PyGithub mplcursors keyboard roifile
# conda install pytables
# pip install pandastable tkcalendar pyarrow XlsxWriter
# conda install markupsafe=2.0.1
# mamba install python=3.8
# mamba install jinja2=3.0
# mamba install spyder=5.1
# pip install allensdk
# setx /M PATH "%PATH%;C:\Users\sp3660\AppData\Roaming\Python\Python38\Scripts"
# mamba install tifffile==2024.4.8 BAD, very slow lots of siseeus 
# caiman timeseries chanche compress= fro compression=
# tifffile issues seem that caiman is working with a different mirror v0.12, but that gives environement problems, doesnt seem to be able to install with conda
# conda install tifffile==0.12 this doesn install 
# pip install tifffile==0.12
# cnat remove tifffile 2022.4.28 because it generate incosnsitencies

 # conda-forge/noarch::nbformat==5.3.0=pyhd8ed1ab_0
 #  - conda-forge/noarch::nbclient==0.6.0=pyhd8ed1ab_0
 #  - conda-forge/noarch::nbconvert-core==6.5.0=pyhd8ed1ab_0
 #  - conda-forge/noarch::nbconvert-pandoc==6.5.0=pyhd8ed1ab_0
 #  - conda-forge/noarch::nbconvert==6.5.0=pyhd8ed1ab_0
 #  - conda-forge/noarch::notebook==6.4.11=pyha770c72_0
 #  - conda-forge/win-64::spyder==5.1.5=py38haa244fe_1
 #  - conda-forge/win-64::widgetsnbextension==3.6.0=py38haa244fe_0
 #  - conda-forge/noarch::ipywidgets==7.7.0=pyhd8ed1ab_0
 #  - conda-forge/win-64::jupyter==1.0.0=py38haa244fe_7




#         to do configure gitpyhton
#     conda install -c conda-forge opencv
#     conda config --set allow_conda_downgrades true
#     conda install -n root conda=4.6
#     conda config --set auto_update_conda false
#     conda config --remove channels conda-forge
    
# linux caiman 
# git clone https://github.com/flatironinstitute/CaImAn
# cd CaImAn/
# mamba env create -f environment.yml -n caiman
# source activate caiman
# pip install -e .    
    
    
    

# conda config --add channels conda-forge


# how to sign in to github wit acces token

# synchinhe medneley remark on smal laptop
#pyhton=3.9
# conda activate remark
# cd Documents\Github\mendeley...\
# pyhton sync.py


# """
from pathlib import Path
import tkinter as tk
from sys import platform
import socket
from project_manager.ProjectManager import ProjectManager
import urllib3
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
# lazy_import
from pprint import pprint
from project_manager.todo_Selection import TODO_Selection


house_PC='DESKTOP-V1MT0U5'
lab_PC='DESKTOP-OKLQSQS'
small_laptop_ubuntu='samuel-XPS-13-9380'
small_laptop_kali='samuel-XPS-13-9380'
big_laptop_ubuntu='samuel-XPS-15-9560'
big_laptop_arch='samuel-XPS-15-9560'

if platform == "win32":
    if socket.gethostname()==house_PC:
        githubtoken_path=r'C:\Users\Samuel\Documents\Github\GitHubToken.txt'
        computer=house_PC
    elif socket.gethostname()==lab_PC:
        githubtoken_path=r'C:\Users\sp3660\Documents\Github\GitHubToken.txt'
        computer=lab_PC
        
elif platform == "linux" or platform == "linux2":
    if socket.gethostname()==small_laptop_ubuntu:
        computer=small_laptop_ubuntu
        githubtoken_path='/home/samuel/Documents/Github/GitHubToken.txt'
        # Path('/home/samuel/Documents/Github/GitHubToken.txt')
        print('TO DO')



"""
# current errors and excepions

    anotate how to add a new 

    SPHQ ignored when loading mmaps 
    SPJO/P day 26 had corrupted csv files I CREATED EMPTY VOLTAGE FILES
    
    20220217  tMPO{ORARILY BLOCK PROCESSING OF HAKIM FOLDE ACQUISITIOON UNTIL AFTER I ADD SCANIMAGE by adding a function if hakim in
    
    
    ALL MY PLASMIDS ARE IN THE DNA FRIDGE IN BOX AT THE BOTTOM WITH ORANGE TAPE
    
    to do 
        check new breeding corretc cages
        add all genoitypes to code
    
    
    
    mendeley transfer
    smal laptop
        conda activate remark
        cd Documents/Github/mendeley-rMsync/
        pyhton3 sync.py
    
    bash scripts for cleaning up repos
    git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  sed -n 's/^blob //p' |
  sort --numeric-sort --key=2 |
  cut -c 1-12,41- |
  $(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest


git filter-branch --index-filter "git rm -rf --cached --ignore-unmatch   boc/cell_specimens.json" HEAD;
git for-each-ref --format="%(refname)" refs/original/ | while read ref; do git update-ref -d $ref; done





git reflog expire --expire=now --all && git gc --prune=now --aggressive



    
    
"""


# host="8.8.8.8"
# port=53
# timeout=1000
# try:
#     socket.setdefaulttimeout(timeout)
#     socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
#     print('internet')
# except socket.error as ex:
#     print(ex)
#%% select analysys
projectManager=ProjectManager(githubtoken_path, computer, platform)
# ProjectManager.clone_project('LabNY')   
#%%
selection=TODO_Selection(projectManager)
selection.mainloop()


#%%


lab, allen, MouseDat, datamanaging,session_name, mousename, mouse_object, allacqs, selectedaqposition, acq, analysis,full_data=selection.to_return






#%%
   
# testiing for negative correlation\ and stuff, compare jesus method with standard correlation and beware of the differences
# from jesusMiscFunc import get_significant_network_from_raster, filter_raster_by_network, find_peaks, change_raster_bin_size, get_peak_indices, get_adjacency_from_raster
# from scipy.stats import zscore


# simpletest=np.array(([1 ,1 ,1, 1 ,0, 0, 0, 0],[0, 0, 0, 0, 1, 1, 1, 1], [1 ,1 ,1, 1 ,0, 0, 0, 0],[0, 1, 0, 0, 1, 1, 0, 1],[1 ,0 ,0, 1 ,1, 0, 0, 1],[0, 0, 1, 0, 0, 0, 1, 1]))

# mat=get_adjacency_from_raster(simpletest)
# matp=np.corrcoef(simpletest)
# matj=get_adjacency_from_raster(simpletest,'jaccard')

# distance = squareform(pdist(simpletest, 'cosine'))
# distancec = squareform(pdist(simpletest, 'correlation'))
# distancej = squareform(pdist(simpletest, 'jaccard'))


# simpletest2=np.array(([1 ,1 ,1, 1 ,-1, -1, -1,-1],[-1, -1, -1,-1, 1, 1, 1, 1]))

# mat2=get_adjacency_from_raster(simpletest2 ,)
# matp2=get_adjacency_from_raster(simpletest2,'pearson')
# matj2=get_adjacency_from_raster(simpletest2,'jaccard')

# distance2 = squareform(pdist(simpletest2, 'cosine'))
# distancec2 = squareform(pdist(simpletest2, 'correlation'))
# distancej2 = squareform(pdist(simpletest2, 'jaccard'))





# mat=get_adjacency_from_raster(jesusres_object.analysis['Raster'] ,)
# matp=np.corrcoef(jesusres_object.analysis['Raster'])*(1-np.eye(jesusres_object.analysis['Raster'].shape[0]))
# fig, ax1 = plt.subplots(1)
# pos=ax1.imshow(matp, cmap='jet', )
# fig.colorbar(pos, ax=ax1)


# scoremat=zscore(matp)
# fig, ax1 = plt.subplots(1)
# pos=ax1.imshow(scoremat, cmap='jet', )
# fig.colorbar(pos, ax=ax1)
# from scipy.spatial.distance import squareform, pdist



# distance = squareform(pdist(jesusres_object.analysis['Raster'], 'cosine'))
   
# sim = 1-distance
# fig, ax1 = plt.subplots(1)
# pos=ax1.imshow(sim, cmap='jet', )
# fig.colorbar(pos, ax=ax1)

# fig, ax = plt.subplots(1)
# N, bins, patches = ax.hist(matp.flatten(), bins=50) #initial color of all bins


# test=np.histogram(sim)


   



#%%  github to do
# #%% git managing TO WORK ON IT

# # labRepo=lab.repo_object

# # assert not labRepo.is_dirty()  # check the dirty state
# # labRepo.untracked_files  


# # assert os.path.isdir(labRepo.working_tree_dir)                   # directory with your work files
# # assert labRepo.git_dir.startswith(labRepo.working_tree_dir)  # directory containing the git repository

#               # the commit pointed to by head called master
# index = labRepo.index
# count_modified_files = len(labRepo.index.diff(None))
# count_staged_files = len(labRepo.index.diff("HEAD"))
# commits = list(labRepo.iter_commits('master'))
#%%
