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



# SET GITHUB TOKEN PATH
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


git filter-branch --index-filter "git rm -rf --cached --ignore-unmatch   ny_lab/TestOthers/desktop_cam_20220614-161717.avi" HEAD;
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
#%% initialize the project manager
projectManager=ProjectManager(githubtoken_path, computer, platform)
# ProjectManager.clone_project('LabNY')   
#%%
selection=TODO_Selection(projectManager)
selection.mainloop()

#%%
lab, allen, MouseDat, datamanaging,session_name, mousename, mouse_object, allacqs, selectedaqposition, acq, analysis,full_data=selection.to_return


#%%  github to do
projectManager.check_project_git_status(projectManager)
projectManager.stage_commit_and_push(projectManager)
projectManager.pull_from_github(projectManager)


