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

synchinhe medneley remark on smal laptop
conda activate remark
cd Documents\Github\mendeley...\
pyhton sync.py


"""
from pathlib import Path
import tkinter as tk
from sys import platform
import socket
from project_manager.ProjectManager import ProjectManager
import urllib3
import os
import pandas as pd
from pprint import pprint


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
#%%
projectManager=ProjectManager(githubtoken_path, computer, platform)
# ProjectManager.clone_project('LabNY')   
allen=0
testing=0
data_analysis_database=0
data_analysis_not_yet_database=0
data_processing=0
gui=0

#%%
if testing:
  gui=0
  data_analysis_database=0
  data_processing=0
  data_analysis_not_yet_database=0

  
if data_analysis_database:
  gui=0
  testing=0
  data_processing=0
  data_analysis_not_yet_database=0

if data_processing:
  gui=0
  testing=0
  data_analysis_database=0
  data_analysis_not_yet_database=0
  
if data_analysis_not_yet_database:
  gui=0
  testing=0
  data_analysis_database=0
  data_processing=0



if allen:
    testing=0
    gui=0
    allen=projectManager.initialize_a_project('AllenBrainObservatory',gui)   
    #%%
    # allen.testing()

else:

    update=0
    lab=projectManager.initialize_a_project('LabNY', gui)   
   
    
    if not gui:
        MouseDat=lab.database
        lab.do_datamanaging()
        datamanaging=lab.datamanaging
        if update:
            # this was done for the disk change i think
            # datamanaging.update_pre_process_slow_data_structure(update=True)
            datamanaging.update_all_imaging_data_paths()
            datamanaging. read_all_data_path_structures()
            # datamanaging.delete_pre_procesed_strucutre_mouse_without_data()
            # datamanaging.read_all_imaging_sessions_from_directories()
            
            datamanaging.read_all_immaging_session_not_in_database()
  
#%%  
if data_analysis_database:
    #%selecting an acquisition
    pass
    mousename='SPKG'
    mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
    allacqs=mouse_object.all_mouse_acquisitions

    pprint(list(allacqs.keys()))
    # selectedaqposition = int(input('Choose Aq position.\n'))
    selectedaqposition=1
    
    #% getting acq
    acq=allacqs[list(allacqs.keys())[selectedaqposition]]
    selected_acquisition=''
    
    
    # % basic analysis routine
    acq.load_results_analysis() 
    
    #%
    # acq.analysis_object.signals_object.process_all_signals()
    acq.analysis_object.signal_alignment_testing()
    #%
    
    acq.analysis_object.check_all_jesus_results()
    pprint(acq.analysis_object.jesus_results_list)
    #%
    import numpy as np
    import matplotlib.pyplot as plt
    i=0
    deconv='MCMC'
    deconv='dfdt'
    all_divisions=0
    index=0
    
    if all_divisions:
        list_to_it=acq.analysis_object.jesus_results_list
    else:
        list_to_it=[acq.analysis_object.jesus_results_list[index]]
    
    for j in list_to_it:
        resultstoload=i
        if deconv in j:
            path= j
            print(j)
            acq.analysis_object.load_jesus_results( path)
            acq.analysis_object.jesus_runs
            jesusres_object= acq.analysis_object.jesus_runs[list( acq.analysis_object.jesus_runs.keys())[i]]    
            jesusres_object.load_analysis_from_file()
        
            jesusres_object.plotting_summary()
            jesusres_object.analysis['Ensembles']['StructureWeightsSignificant']
            
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span t
            ax.imshow(jesusres_object.analysis['Ensembles']['StructureWeightsSignificant'], aspect='auto')
            ax.set_xlabel('Cell')
            fig.supylabel('Ensemble Number')
            fig.suptitle('Ensemble Cell Weights')
            
            plt.show()

            ens=jesusres_object.analysis['Ensembles']['StructureWeightsSignificant']
            print(len(ens))
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span th
            ax.plot( ens[0,:])
            ax.set_xlabel('Cell')
            fig.supylabel('Ensemble One')
            fig.suptitle('Ensemble Cell Weights')
            plt.show()

            ense1index=np.where(ens[0,:]>0.05)[0]
            
            
            activ=acq.analysis_object.separated_planes_combined_gratings_binarized_dfdt[list(acq.analysis_object.separated_planes_combined_gratings_binarized_dfdt.keys())[0]]
            
            acq.analysis_object.resampled_stim_matrix
              
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span th
            ax.imshow(activ[ense1index,:], aspect='auto')
            ax.set_xlabel('Frame')
            fig.supylabel('Ensemble One Cells')
            fig.suptitle('Ensemble 1 Activity')
            plt.show()

            
            start=20000
            end=21000
                
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span th
            ax.plot(activ[ense1index[2],start:end])
            ax.plot( acq.analysis_object.resampled_stim_matrix[start:end])
            ax.set_xlabel('Frame')
            fig.supylabel('Ensemble One FIrstCell')
            fig.suptitle('Ensemble 1 Activity')
            
            plt.show()
            i=i+1
   
 
    from jesusMiscFunc import get_significant_network_from_raster, filter_raster_by_network, find_peaks, change_raster_bin_size, get_peak_indices, get_adjacency_from_raster
    from scipy.stats import zscore
    
    
    simpletest=np.array(([1 ,1 ,1, 1 ,0, 0, 0, 0],[0, 0, 0, 0, 1, 1, 1, 1], [1 ,1 ,1, 1 ,0, 0, 0, 0],[0, 1, 0, 0, 1, 1, 0, 1],[1 ,0 ,0, 1 ,1, 0, 0, 1],[0, 0, 1, 0, 0, 0, 1, 1]))

    mat=get_adjacency_from_raster(simpletest)
    matp=np.corrcoef(simpletest)
    matj=get_adjacency_from_raster(simpletest,'jaccard')

    distance = squareform(pdist(simpletest, 'cosine'))
    distancec = squareform(pdist(simpletest, 'correlation'))
    distancej = squareform(pdist(simpletest, 'jaccard'))


    simpletest2=np.array(([1 ,1 ,1, 1 ,-1, -1, -1,-1],[-1, -1, -1,-1, 1, 1, 1, 1]))
    
    mat2=get_adjacency_from_raster(simpletest2 ,)
    matp2=get_adjacency_from_raster(simpletest2,'pearson')
    matj2=get_adjacency_from_raster(simpletest2,'jaccard')

    distance2 = squareform(pdist(simpletest2, 'cosine'))
    distancec2 = squareform(pdist(simpletest2, 'correlation'))
    distancej2 = squareform(pdist(simpletest2, 'jaccard'))

    
    
    
    
    mat=get_adjacency_from_raster(jesusres_object.analysis['Raster'] ,)
    matp=np.corrcoef(jesusres_object.analysis['Raster'])*(1-np.eye(jesusres_object.analysis['Raster'].shape[0]))
    fig, ax1 = plt.subplots(1)
    pos=ax1.imshow(matp, cmap='jet', )
    fig.colorbar(pos, ax=ax1)
    

    scoremat=zscore(matp)
    fig, ax1 = plt.subplots(1)
    pos=ax1.imshow(scoremat, cmap='jet', )
    fig.colorbar(pos, ax=ax1)
    from scipy.spatial.distance import squareform, pdist

    

    distance = squareform(pdist(jesusres_object.analysis['Raster'], 'cosine'))
   
    sim = 1-distance
    fig, ax1 = plt.subplots(1)
    pos=ax1.imshow(sim, cmap='jet', )
    fig.colorbar(pos, ax=ax1)
    
    fig, ax = plt.subplots(1)
    N, bins, patches = ax.hist(matp.flatten(), bins=50) #initial color of all bins

    
    test=np.histogram(sim)
#%esnembk\
    ensemble1drifitng=[ 0,  1,  2,  4,  6,  7,  8,  9, 11, 14, 15, 18, 19, 20, 21, 34, 37,62, 77, 83]
    


    
    ensemble1full=[  0,  1,  2,   6,   7,   8,  11,  13,  15,  18,  20,  21,  47, 62,  67,  75,  83,  89,  93, 114]

   

    intersection = set(ensemble1drifitng). intersection(ensemble1full) 
    unshared=set(ensemble1drifitng).symmetric_difference(ensemble1full)
    
    unshaered_drif=set(ensemble1drifitng). intersection(unshared) 
    unshaered_full=set(ensemble1full). intersection(unshared) 
    

    len(intersection)/len(ensemble1drifitng)
    len(intersection)/len(ensemble1full)

    
    
    
    #%
    
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane1', segment='DriftingGratings')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane2', segment='DriftingGratings')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane3', segment='DriftingGratings')

    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane1', segment='Full')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane2', segment='Full')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='MCMC', plane='Plane3', segment='Full')
    
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane1', segment='DriftingGratings')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane2', segment='DriftingGratings')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane3', segment='DriftingGratings')

    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane1', segment='Full')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane2', segment='Full')
    # acq.analysis_object.load_jesus_analysis(binary_raster_to_proces='dfdt', plane='Plane3', segment='Full')



    # test=acq.analysis_object.jesus_runs
    # test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][-1]=test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][-1].analysis
    # tdictosave={}
    # tdictosave['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'+'_'+
    #            test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][0]+'_'+
    #       test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][1]+'_'+ 
    #       test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][2]]=[ test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][0:-2], test['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'][-1]]
    
    # with open( acq.analysis_object.jesus_runs_path, 'wb') as f:
    #     # Pickle the 'data' dictionary using the high`est protocol available.
    #     pickle.dump(tdictosave, f, pickle.HIGHEST_PROTOCOL)
    # # acq.analysis_object.identify_in_pyr()
    
    # if os.path.isfile( acq.analysis_object.jesus_runs_path):
    #     with open(  acq.analysis_object.jesus_runs_path, 'rb') as file:
    #         jesus_runs= pickle.load(file)
    # if os.path.isfile( pth):
    #     with open(  pth, 'rb') as file:
    #         jesus_runs= pickle.load(file)
    
elif data_analysis_not_yet_database:
#% loading session nt yet in dab
    session_name='20220213'
    mousename='SPJZ'
    mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
    
    datamanaging.all_existing_sessions_not_database_objects[session_name].read_all_yet_to_database_mice()
    
    allacqs=mouse_object.all_mouse_acquisitions
    pprint(list(allacqs.keys()))
    # selectedaqposition = int(input('Choose Aq position.\n'))
    selectedaqposition=1
    
    #% getting acq
    acq=allacqs[list(allacqs.keys())[selectedaqposition]]
    #%
    acq.load_results_analysis()
    
    #%
    # acq.analysis_object.signals_object.process_all_signals()
    acq.analysis_object.signal_alignment_testing()
    acq.analysis_object.some_ploting()
    acq.analysis_object.signals_object.plot_all_basics()
    acq.analysis_object.signals_object.plot_processed_allen()
    #%
    
    acq.analysis_object.check_all_jesus_results()
    pprint(acq.analysis_object.jesus_results_list)


    
    
    
elif data_processing:
    #%% reprocesing session already in database
    prairie_session=datamanaging.all_existing_sessions_database_objects['20210520']
    # this is the celan up and org, this has to be done first
    prairie_session.process_all_imaged_mice()
    

   
    
    #%% get dataset and add raw path to redo inital preprocesing
    # acq.all_datasets
    # dataset=list(acq.all_datasets.items())[0]
    # dat= dataset[1]
    # dat.metadata.acquisition_metadata['AcquisitonRawPath']
    # channel=os.path.split(dat.selected_dataset_mmap_path)[1]
    # if 'Green' in channel:
    #     rawchan='Ch2Green'
    # else:
    #     rawchan='Ch1Red'
    # dataset_name=os.path.split(os.path.split(os.path.split(os.path.split(dat.selected_dataset_mmap_path)[0])[0])[0])[1]
    # dat.selected_dataset_raw_path=os.path.join(dat.metadata.acquisition_metadata['AcquisitonRawPath'],dataset_name, rawchan, dat.plane.lower())
    # dat.do_bidishift(force=True)
    



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
    
elif testing:
    
    
    
    
    pass
