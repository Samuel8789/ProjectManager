# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:27:59 2021

@author: sp3660
requirements
check environemnet and anconda
install anaconda

check gtihub token
git
numpy
matplotlib
    pip install GitPython
        to do configure gitpyhton
    pip install PyGithub
    conda create -n caiman
    conda activate caiman
    conda install -c conda-forge opencv
    conda install mpl cursors
    pip install pandastable
    pip install tkcalendar
    conda config --set allow_conda_downgrades true
    conda install -n root conda=4.6
    conda config --set auto_update_conda false
    conda config --remove channels conda-forge
    pip install allensdk
    
linux caiman 
git clone https://github.com/flatironinstitute/CaImAn
cd CaImAn/
mamba env create -f environment.yml -n caiman
source activate caiman
pip install -e .    
    
    
    
conda config --set channel_priority flexible
conda config --add channels conda-forge


how to sign in to github wit acces token

"""
from pathlib import Path
import tkinter as tk
from sys import platform
import socket
from project_manager.ProjectManager import ProjectManager
import urllib3
import os
import pandas as pd



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
    SPHQ ignored when loading mmaps 
    SPJO/P day 26 had corrupted csv files I CREATED EMPTY VOLTAGE FILES
    added two lines to create metdata and reference folder in create dir structure for datasets
    
    20220217  tMPO{ORARILY BLOCK PROCESSING OF HAKIM FOLDE ACQUISITIOON UNTIL AFTER I ADD SCANIMAGE by adding a function if hakim in
    
    right now i have to change the lab paths to insert manually in project manager
    also in lab_ny_run change path

    
    
    ALL MY PLASMIDS ARE IN THE DNA FRIDGE IN BOX AT THE BOTTOM WITH ORANGE TAPE
    
    to do 
        check new breeding corretc cages
        add all genoitypes to code
    
    
    
    mendeley transfer
    smal laptop
        conda activate remark
        cd Documents/Github/mendeley-rMsync/
        pyhton3 sync.py
    
    
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

ProjectManager=ProjectManager(githubtoken_path, computer, platform)
# ProjectManager.clone_project('LabNY')   
allen=0
testing=1
gui=1
if testing:
  gui=0

if allen:
    testing=0
    gui=0
    allen=ProjectManager.initialize_a_project('AllenBrainObservatory',gui)   

#%%
else:

    update=0
    lab=ProjectManager.initialize_a_project('LabNY', gui)   
   
    
    if not gui:
        MouseDat=lab.database
        datamanaging=lab.datamanaging
        if update:
            # datamanaging.update_pre_process_slow_data_structure(update=True)
            datamanaging.update_all_imaging_data_paths()
            #%%
            datamanaging. read_all_data_path_structures()
            # datamanaging.delete_pre_procesed_strucutre_mouse_without_data()
            #%%
            # datamanaging.read_all_imaging_sessions_from_directories()
            
            datamanaging.read_all_immaging_session_not_in_database()
  
if testing:
    
    #%% doing and checking deep caiman
    # mice_list=['SPKJ']
    # datamanaging.do_deep_caiman_of_mice_datasets(mice_list)
    #%%
    # datamanaging.all_deep_caiman_objects
    # caimanobject=datamanaging.all_deep_caiman_objects[list(datamanaging.all_deep_caiman_objects.keys())[0]]
    # caimanobject.load_results_object()
    #%%
    # datamanaging.copy_all_mouse_with_data_to_working_path(['SPJC','SPIN' ,'SPKG','SPIL','SPIK'])
    # caimanobject.CaimanResults_object.open_caiman_sorter()
    # %%preprocessing parairiew before database       
    for i in datamanaging.all_existing_sessions_not_database_objects.values() :
      i.process_all_imaged_mice()
    # for i in datamanaging.all_existing_sessions_database_objects.values() :
    #     i.process_all_imaged_mice()
    
    #%%
    # session_name='20220129'
    # datamanaging.all_existing_sessions_not_database_objects[session_name].load_all_yet_to_database_mice()
    # datamanaging.all_existing_sessions_not_database_objects[session_name].process_all_imaged_mice()

    # mouse_codes=datamanaging.all_existing_sessions_not_database_objects[session_name].session_imaged_mice_codes
    
       
    # imaging_session=mouse_object.imaging_sessions_not_yet_database_objects[session_name]
    # os.startfile(imaging_session.mouse_session_path)
    
    
    # acqs=[datamanaging.all_experimetal_mice_objects[mouse_code].all_mouse_acquisitions  for mouse_code in mouse_codes]
    # imaging_session=[datamanaging.all_experimetal_mice_objects[mouse_code]  for mouse_code in mouse_codes]
    
    
    
    # fullalen=acqs[1][list(acqs[1].keys())[-2]]
    
    
    # fullalen.face_camera.full_eye_camera.play()
    # test=pd.DataFrame(fullalen.voltage_signals_dictionary['Locomotion'])
    # test.plot()
    
    # fullalen.metadata_object
    
    # fullalen.all_datasets
    # surface=list(fullalen.FOV_object.all_datasets[-1].values())[0]
    # surface_green=list(surface.all_datasets.values())[0]
    # surface_red=list(surface.all_datasets.values())[0]
    # surface_green.summary_images_object.plotting()
    # surface_red.summary_images_object.plotting()
    
    
    
    
    # fullalgrenplane1=fullalen.all_datasets[list(fullalen.all_datasets.keys())[0]]
    # fullalgrenplane1.kalman_object.dataset_kalman_caiman_movie.play(fr=1000)
    # fullalgrenplane1.summary_images_object.plotting()
    # # %matplotlib qt
    # fullalgrenplane1.most_updated_caiman.cnm_object.estimates.view_components()
    # fullalgrenplane1.selected_dataset_mmap_path
    # os.startfile(fullalgrenplane1.selected_dataset_mmap_path)
    
    
    # coord0=list(fullalen.FOV_object.mouse_imaging_session_object.all_0coordinate_Aquisitions.values())[0]
    # widef=fullalen.FOV_object.mouse_imaging_session_object.widefield_image[list(fullalen.FOV_object.mouse_imaging_session_object.widefield_image.keys())[0]]
    # widef.plot_image()
    
        #%%
    # datamanaging.all_existing_sessions_not_database_objects['20211111'].process_all_imaged_mice()
    # mice_codes=['SPJA', 'SPJC']
    # datamanaging.copy_all_mouse_with_data_to_working_path(mice_codes)
    
    
    
    # all_prairie_sessions=datamanaging.all_existing_sessions
    # all_database_sessions_objects=datamanaging.all_existing_sessions_database_objects
    # all_database_sessions=datamanaging.all_existing_sessions_database
    
    # all_experimetal_mice_objects=datamanaging.all_experimetal_mice_objects
    # all_imaged_mice_df=datamanaging.all_imaged_mice
    # primary_data_mice_codes=datamanaging.primary_data_mice_codes
    # primary_data_mice_paths=datamanaging.primary_data_mice_paths
    # secondary_data_mice_codes=datamanaging.secondary_data_mice_codes
    # secondary_data_mice_paths=datamanaging.secondary_data_mice_paths
    # secondary_data_mice_projects=datamanaging.secondary_data_mice_projects
    
    
    # selected_mouse='SPJA'
    
    # selected_mouse_info={'Project':secondary_data_mice_projects[selected_mouse], 
    #                  'Path': secondary_data_mice_paths[selected_mouse], 
    #                  'Code':selected_mouse,
    #                  'Mouse_object': all_experimetal_mice_objects[selected_mouse],
    #                  'imaging_sessions':all_experimetal_mice_objects[selected_mouse].imaging_sessions_objects
    #                      }
    # selected_mouse_info['Mouse_object'].get_all_mouse_FOVdata_datasets()
    # mooom=selected_mouse_info['Mouse_object'].all_mouse_FOVdata_datasets
    # session=selected_mouse_info['imaging_sessions']['20210624']  
    # fov=session.all_FOVs['FOV_1']                   
    # fov.all_existing_1050tomato
    # fov.all_existing_1050tomato
    # dataset=fov.all_aquisitions[list(fov.all_aquisitions.keys())[0]]
    
    # '''  
    # 1st clean up the raw folder
    # then add the session to database
    # then trnasform raw tiff to mmap and transfer all other files
       
    #  parairie_session_functions
    # load_all_images_as_mmap this is for a single prairie_imaging_session, it create a mouse imaging session and lead all images as mmaps. I still have to solev the issue with very large multiplanes mostly SPHQ and furture multiplanes, se how to organize in raw and also hot to convert to mmap
    # '''
    
    
    
    #%%  
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
    pass