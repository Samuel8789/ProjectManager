# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:57:50 2022

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter
from pathlib import Path
from sys import platform
import socket
import urllib3
# from pyforest import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import matplotlib as mpl
from pprint import pprint
import scipy as spy
import scipy.io as sio

from scipy.spatial.distance import squareform, pdist
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore
import copy
from scipy import interpolate
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation 
from IPython.display import HTML



mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

#%%
    
class TODO_Selection(tk.Tk):
    def __init__(self, projectManager):
        super().__init__()
        self.title('To DO Select')
        self.geometry('400x400')
        self.projectManager=projectManager
        
        self.things_to_do=[
        # LAB gui 0:6
        'process_imaging_data',
        'add_imaging_data_to_database',
        'do_mouse_visit',
        'prepare_mouse_surgeries',
        'update_mouse_surgeries',
        'plan_mouse_surgeries',
        #AllenBO 6:7
        'work_with_allenBO',
        #Data processing 7:12
        'testing_datamanagin',
        'testing_image_processing',
        'testing_image_analysis',
        'deep_caiman',
        'dataset_focus',
        #Analysis 12:
        'do_data_analysis_from_non_database',
        'do_data_analysis_from_database',
        'explore_analysis',
        'do_visualstim_indexing',
        'do_jesus_ensemble',
        'do_tuning_allen',
        'do_yuriy_ensembles']
        
        self.gui=0

        self.var=tk.StringVar()
        self.selection_cmbbx=Listbox( self, listvariable=self.var, selectmode=MULTIPLE, width=40, height=20, exportselection=0)
        self.selection_cmbbx.grid(column=0, row=3)
        self.var.set(self.things_to_do)

        self.accept_button=ttk.Button( self, text='Work', command=self.do_some_work)
        self.accept_button.grid(row=0, column=3)    

    def do_some_work(self):
        # todo=self.selection_cmbbx.var.get()
        
        self.selected_things_to_do = list()
        selection = self.selection_cmbbx.curselection()
        if selection:
            for i in selection:
                thing_to_do = self.selection_cmbbx.get(i)
                self.selected_things_to_do.append(thing_to_do)
        else:
             self.selected_things_to_do=list(ast.literal_eval( self.var.get()))
        
        
#%% Open lab gui
        if [todo for todo in self.selected_things_to_do if todo in self.things_to_do[0:6]]:

            self.gui=1
            self.destroy() 
            
            lab=self.projectManager.initialize_a_project('LabNY', self.gui) 
            # if 'add_imaging_data_to_database' in self.selected_things_to_do:
            #     # lab.do_datamanaging()
            #     pass

            self.to_return=[lab,[], [],[],[],[], [],[],[],[], [],[]]

#%% ALLEN BO
        elif self.things_to_do[6] in self.selected_things_to_do:
            
            allen=self.projectManager.initialize_a_project('AllenBrainObservatory',self.gui)  
            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            self.to_return=[lab, allen,[],[],[],[], [],[],[],[], [],[]]

            self.destroy()
            return
            allen.set_up_monitor( screen_size=[750,1280], lenght=37.7)
            allen.get_visual_templates()
            # allen.get_gratings()
            # allen.get_drifting_gratings()
        
            allen.get_selection_options()
            
            area='VISp'
            line='Vglut1'
            # line='PV'

            depth=175
            stim='drifting_gratings'
            
            _, _, exp_list_all_sessions, exps_all_sessions=allen.select_lines_and_locations(area=area, line=line, depth=depth)
            exp_containers, containers, exp_list_by_stim, exps_by_stim=allen.select_lines_and_locations(area=area, line=line, depth=depth, stim=stim)
            print(exps_by_stim)
            selection=0
            selected_session_id, selected_container_id=allen.select_experiment(exp_list_by_stim, exp_selection_index=selection)
            exps_list_by_fov, exps_by_fov=allen.select_exp_from_imaged_fov(exp_containers=exp_containers, container_selection_index=containers[containers['id']==selected_container_id].index[0])

            print(containers)
            print(exps_by_fov)
            data_set, deconvolved_spikes, session_stimuli= allen.download_single_imaging_session_nwb(selected_session_id)

            all_exp_fov=[]
            for i in exps_list_by_fov:
                all_exp_fov.append(allen.download_single_imaging_session_nwb(i['id']))

            filename='Allen_{}_{}_{}_{}'.format(line, area, depth, selected_container_id )
            for i in range(3):
                print(all_exp_fov[i][-1])
            
            selection=1
            data_set=all_exp_fov[selection][0]
            spikes=all_exp_fov[selection][1]
            plt.imshow(spikes, cmap='binary', aspect='auto', vmax=0.01)
            
            from ny_lab.data_analysis.resultsAnalysis import ResultsAnalysis
            allen_results_analysis=ResultsAnalysis(allen_BO_tuple=(allen,data_set,spikes ))
            allen_results_analysis.do_PCA( allen_results_analysis.drifting[0].mean_sweep_response, allen_results_analysis.drifting[0].sweep_response, allen_results_analysis.drifting[-1])
            
            
#%% data processing
        elif [todo for todo in self.selected_things_to_do if todo in self.things_to_do[7:12]] : 

            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            MouseDat=lab.database
            lab.do_datamanaging()
            datamanaging=lab.datamanaging
            
            self.to_return=[lab, [],MouseDat,datamanaging,[],[], [],[],[],[], [],[]]
#%% testing datamanaging

            if 'testing_datamanagin' in self.selected_things_to_do:

                self.destroy()
                return


                
                # this was done for the disk change i think
                # datamanaging.update_pre_process_slow_data_structure(update=True)
                datamanaging.update_all_imaging_data_paths()
                datamanaging. read_all_data_path_structures()
                # datamanaging.delete_pre_procesed_strucutre_mouse_without_data()
                # datamanaging.read_all_imaging_sessions_from_directories()
                
                datamanaging.read_all_immaging_session_not_in_database()
                
                
               
#%% deep caiman  
            elif 'deep_caiman' in self.selected_things_to_do:
                
                self.destroy()
                return
      
                datamanaging.get_all_deep_caiman_objects()

                datamanaging.all_deep_caiman_objects
                allimagedmice=datamanaging.all_imaged_mice['Code'].unique().tolist()
                
                # tododeepcaiman=['SPHV', 'SPHW','SPHX','SPJB','SPJD','SPKF','SPKH','SPKI','SPKL','SPIG','SPIH']
                tododeepcaiman=['SPGT', 'SPHQ','SPIB','SPIC','SPIL','SPIM','SPIN','SPJF','SPJG','SPJH','SPJI','SPJZ','SPKC','SPKS',	'SPKU'	,'SPKV'	,'SPLE',	'SPLF']
                for i in tododeepcaiman:
                    datamanaging.do_deep_caiman_of_mice_datasets([i])
#%% dataset focus
            elif 'dataset_focus' in self.selected_things_to_do:
                self.destroy()
                return
                
                mousenames=['SPKG']
                mousename=mousenames[0]

                mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                allacqs=mouse_object.all_mouse_acquisitions

                pprint(list(allacqs.keys()))
                # selectedaqposition = int(input('Choose Aq position.\n'))
                selectedaqposition=5
                
                #% getting acq
                acq=allacqs[list(allacqs.keys())[selectedaqposition]]
                acq.get_all_database_info()
                
                alldtsets=acq.all_datasets
                
                pprint(list(alldtsets.keys()))

                selecteddtsetposition=0
                
                #% getting acq
                dtset=alldtsets[list(alldtsets.keys())[selecteddtsetposition]]
                dtset.most_updated_caiman.load_cnmf_object()
                cnm=dtset.most_updated_caiman.cnm_object
                mov=cnm.estimates.A[:,:].toarray()@(cnm.estimates.C+cnm.estimates.YrA)+cnm.estimates.b@cnm.estimates.f
                movob=cm.movie(mov.T.reshape((64415,256,256)))
               
                
                new_param_dict={'nb':3}
                dtset.do_deep_caiman(new_param_dict)
                dtset.galois_caiman(new_param_dict)
                
                
                import caiman as cm
                import os

                plane1=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPKG\imaging\20211015\data aquisitions\FOV_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\planes\Plane1\Green\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64415_.mmap'
                plane2=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPKG\imaging\20211015\data aquisitions\FOV_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\planes\Plane2\Green\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64415_.mmap'
                plane3=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPKG\imaging\20211015\data aquisitions\FOV_1\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000\planes\Plane3\Green\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64415_.mmap'
                planes=[plane1 ,plane2, plane3]
                dirp=r'C:\Users\sp3660\Desktop'

                for i,plane in enumerate(planes):
                    mov=cm.load(plane)
                    mov.save(os.path.join(dirp,'Plane'+str(i)+'.tiff'))
                    
#%% image processing    
            elif 'testing_image_processing' in self.selected_things_to_do:    
                self.destroy()
                return
                sessionnames=['20220223','20220306','20220314','20220331','20220414']
                
                for session_name in sessionnames:

                    prairie_session=datamanaging.all_existing_sessions_database_objects[session_name]
                # this is the celan up and org, this has to be done first
                    prairie_session.process_all_imaged_mice()      
                    
#%% data analysis                   
        elif [todo for todo in self.selected_things_to_do if todo in self.things_to_do[12:]] :   
            
            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            MouseDat=lab.database
            lab.do_datamanaging()
            datamanaging=lab.datamanaging
            
            self.to_return=[lab, [],MouseDat,datamanaging,[],[], [],[],[],[], [],[]]
                    
#%%loading session nt yet in dab
            if 'do_data_analysis_from_non_database' in self.selected_things_to_do:
                session_name='20220330'
                mousename='SPKU'
                mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                
                datamanaging.all_existing_sessions_not_database_objects[session_name].read_all_yet_to_database_mice()
                
                allacqs=mouse_object.all_mouse_acquisitions
                pprint(list(allacqs.keys()))
                # selectedaqposition = int(input('Choose Aq position.\n'))
                selectedaqposition=1
                
                #% getting acq
                acq=allacqs[list(allacqs.keys())[selectedaqposition]]
                #%
                acq.load_results_analysis(nondatabase=True)
                analysis=acq.analysis_object
                self.to_return.extend([session_name,mousename,  mouse_object , allacqs, selectedaqposition, acq, analysis,[]])
                
                self.destroy()
                return

                
                
                #%
                acq.voltage_signal_object.plot_all_signals()
                csvsignals=acq.voltage_signal_object.extraction_object.voltage_signals
                daqsignals=acq.voltage_signal_object.extraction_object.voltage_signals_daq
                
                fig,ax=plt.subplots(1)
                ax.plot(daqsignals['AcqTrig']['AcqTrig'], label='daq')
                ax.plot(csvsignals['AcqTrig'].iloc[:,0], label='prairie')
                ax.legend()
                

                
                difdaq=np.diff(daqsignals['LED']['LED'])
                difprar=np.diff(csvsignals['LED'].iloc[:,0]) 
                mediandaq=sg.medfilt(difdaq, kernel_size=3)
                medianprair=sg.medfilt(difprar, kernel_size=3)
                roundeddaq=np.round(mediandaq)
                roundedprair=np.round(medianprair)
                transitionsdaq=np.where(abs(difdaq)>4)[0]
                transitionsprair=np.where(abs(difprar)>4)[0]
                
                fig,ax=plt.subplots(1)
                ax.plot(daqx, daqsignals['LED']['LED'], label='daq')
                ax.plot(daqx[1:],difdaq, label='diffdaq')
                ax.plot(daqx,csvsignals['LED'].iloc[:,0], label='prairie')
                ax.plot(daqx[1:], difprar, label='diffprair')
                ax.legend()

                
                
                lag=transitionsdaq-transitionsprair
                
                daqx=np.arange(daqsignals['LED']['LED'].size)
                prairirex=daqx+stats.mode(lag)[0][0]

               
                
               

                fig,ax=plt.subplots(1)
                ax.plot(daqx,daqsignals['LED']['LED'], label='daq')
                ax.plot(daqx[1:],difdaq, label='diffdaq')
                ax.plot(prairirex,csvsignals['LED'].iloc[:,0], label='prairie')
                ax.plot(prairirex[1:], difprar, label='diffprair')
               
                ax.legend()
                

                fig,ax=plt.subplots(1)
                ax.plot(daqx,daqsignals['LED']['LED'], label='daq')
                ax.plot(prairirex,csvsignals['LED'].iloc[:,0], label='prairie')
                ax.legend()
                
                
                fig,ax=plt.subplots(1)
                ax.plot(daqx[1:],difdaq, label='diffdaq')
                ax.plot(prairirex[1:], difprar, label='diffprair')
                ax.legend()

                fig,ax=plt.subplots(1)
                ax.plot(daqx,daqsignals['Locomotion']['Locomotion'], label='daq')
                ax.plot(prairirex,csvsignals['Locomotion'].iloc[:,0], label='prairie')               
                ax.legend()

                fig,ax=plt.subplots(1)
                ax.plot(daqx,daqsignals['PhotoTrig']['PhotoTrig'], label='daq')
                ax.plot(prairirex,csvsignals['PhotoTrig'].iloc[:,0], label='prairie')               
                ax.legend()
                
                fig,ax=plt.subplots(1)
                ax.plot(daqx,daqsignals['VisStim']['VisStim'], label='daq')
                ax.plot(prairirex,csvsignals['VisStim'].iloc[:,0], label='prairie')               
                
                ax.plot(daqx,daqsignals['LED']['LED'], label='daq')
                ax.plot(prairirex,csvsignals['LED'].iloc[:,0], label='prairie')           
                ax.legend()
                
                
                
                bidi=datset.bidishift_object
                mov=bidi.shifted_movie
                
                m_mean = mov.mean(axis=(1, 2))
                x=np.arange(len(m_mean))
                dif=np.diff(m_mean)
                median=sg.medfilt(dif, kernel_size=3)
                rounded=np.round(median)
                transitions=np.where(abs(dif)>1000)[0]
                transitions_median=np.where(abs(median)>20)[0]
                transitions_medina_rounded=np.where(abs(rounded)>20)[0]
                led_frame_start=transitions[3]+1
                led_frame_end=transitions[4]+1
                noled=mov[start:end,:,:]
                noled_extended=mov[start-1:end+1,:,:]
                noled_mean = noled.mean(axis=(1, 2))
                noledx=np.arange(len(noled_mean))
                noleddif=np.diff(noled_mean)
                
                
                fig,ax=plt.subplots(1)
                ax.plot(daqx,daqsignals['LED']['LED'], label='daqLED')
                ax.plot(prairirex,csvsignals['LED'].iloc[:,0], label='prairieLED')
                ax.plot(daqx, daqsignals['AcqTrig']['AcqTrig'], label='daq')
                ax.plot(prairirex, csvsignals['AcqTrig'].iloc[:,0], label='prairie')
                ax.legend()
                
                csvsignals['AcqTrig'].iloc[:,0]
                
                difstarttrig=np.diff(csvsignals['AcqTrig'].iloc[:,0])
                median=sg.medfilt(difstarttrig, kernel_size=3)
                rounded=np.round(median)
                transitions=np.where(abs(difstarttrig)>2)[0]
                transitions=transitions+stats.mode(lag)[0][0]

                
                fig,ax=plt.subplots(1)
                ax.plot(daqsignals['LED']['LED'][9230:621718], label='mena signal')
                # ax.plot(x,resampledled, label='mena signal')
                ax.legend()
                
                fig,ax=plt.subplots(1)
                ax.plot(stats.zscore(m_mean[142-1:17387+2]), label='mean signal')
                ax.plot(stats.zscore(resampledled), label='voltage')
 
                ax.legend()
                
                fig,ax=plt.subplots(1)
                ax.plot(stats.zscore(m_mean), label='mean signal')
 
                ax.legend()
                
                resampledled=resample_2(daqsignals['LED']['LED'][10234-1:620716+2],1000/acq.metadata_object.translated_imaging_metadata['FinalFrequency'])
                
                
                def resample_2(x, factor, kind='linear'):
                    n = int(np.ceil(x.size / factor))
                    f = interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
                    return f(np.linspace(0, 1, n)) 

                # acq.analysis_object.signals_object.process_all_signals()
                # analysis.signal_alignment_testing()
                # analysis.some_ploting()
                # analysis.signals_object.plot_all_basics()
                # analysis.signals_object.plot_processed_allen()
                # #%
                
                # analysis.check_all_jesus_results()
                # pprint(acq.analysis_object.jesus_results_list)
                
                
            

                                    

#%% analysis form database 
            elif 'do_data_analysis_from_database' in self.selected_things_to_do:
                
                # select mouse
                mousenamesinter=['SPKG','SPHV']
                mousenameschand=['SPKQ','SPKS','SPJZ','SPJF','SPJG']
                
                #spKQ chandeliers plane1 18, 27, 121, 228
                #spKQ chandeliers plane2 52(fist pass caiman)

                # mousename=mousenameschand[0]
                mousename=mousenamesinter[0]

                mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                
                # select aquisition
                allacqs=mouse_object.all_mouse_acquisitions
                pprint(list(allacqs.keys()))
                # selectedaqposition = int(input('Choose Aq position.\n'))
                selectedaqposition=1
                acq=allacqs[list(allacqs.keys())[selectedaqposition]]
                
                # get datasets
                acq.get_all_database_info()
                acq.load_results_analysis(new_full_data=False) 
                # acq.load_results_analysis(new_full_data=True) 
                analysis=acq.analysis_object
                full_data=analysis.full_data
            
                
                self.to_return[5:]=[ mousename,  mouse_object , allacqs, selectedaqposition, acq, analysis,full_data]
#%% analysis exploration 

                if 'explore_analysis'  in self.selected_things_to_do:
                    
                      
                    self.destroy()
                    return
                    # %%VIARIABLE SELECTONS
                    trace_types=['demixed', 'denoised', 
                     'dfdt_raw',  'dfdt_smoothed', 'dfdt_binary',
                     'foopsi_raw', 'foopsi_smoothed','foopsi_binary',
                     'mcmc_raw','mcmc_smoothed','mcmc_binary']
                    trace_type=trace_types[4]
                    final_binary_trace_types=['dfdt_binary','mcmc_binary']
    
                    
                    paradigms=['Movie1','Spontaneous','Drifting_Gratings','Movie3']
                    paradigm=paradigms[2]
    
                    planes=['All_planes_rough', 'Plane1', 'Plane2', 'Plane3']
                    plane=planes[0]
                    
                    full_raster_pyhton_cell_idx=252
    
                    matlab_sorter_plane='Plane3'
                    matlab_sorter_idx=136
                    
                    full_raster_pyhton_cell_idx_list=[2,12,54,123,201,305]
    
                    selected_cells_options=['All','Pyramidal','Interneurons',full_raster_pyhton_cell_idx_list]
                    selected_cells=selected_cells_options[0]
                    
                    directions=np.linspace(0,360-45,8).astype('uint16')
                    orientations=np.linspace(0,180-(180/4),4).astype('uint16')
                    temp_frequencies=np.array([1,2,4,8,15]).astype('uint16')
                    direction_list=[0,45,90,135,180,270,315]
                    temporal_frequency_list=np.array(temp_frequencies[:])
                
                    dur_frames=[33,50]
                    isi_frames=[16,25]
                    selected_dur_frames=dur_frames[0]
                    selected_isi_frames=isi_frames[0]
                    
                    drifting_options=[direction_list,temporal_frequency_list, selected_dur_frames,selected_isi_frames]
    
    
                    #revariabling
                    full_imaging_data=full_data['imaging_data']
                    vistsiminfo=full_data['visstim_info']
                    meta=analysis.metadata_object
                    pyr_int_ids_and_indexes=analysis.pyr_int_ids_and_indexes
                    pyr_int_identification=analysis.pyr_int_identification
                    selected_plane_pyhton_sorter_cell_ids=full_imaging_data[plane]['CellIds']
                    #%% SLICING AND INDEXING CELLS STIMULI PLANES AND TRACES EXAMPLES
                   
                    
                    #get info of a cell indexed from a python raster
                    if full_raster_pyhton_cell_idx:
                        selected_plane, total_cells,\
                            full_raster_cell_python_idx, single_plane_selected_pyhton_cell_idx,single_plane_sorter_pyhton_idx,matlab_sorter_cell_id\
                                =analysis.convert_full_planes_idx_to_single_plane_final_indx(full_raster_pyhton_cell_idx,plane)
                                
                                
                     #get info of a cell indexed from a the matlab sorter
                    # if matlab_sorter_idx:
                    #     selected_plane, total_cells,\
                    #     full_raster_cell_python_idx, single_plane_selected_pyhton_cell_idx, single_plane_sorter_pyhton_idx, matlab_sorter_cell_id\
                    #         =analysis.convert_full_planes_idx_to_single_plane_final_indx\
                    #             (analysis.get_full_raster_indx_from_matlab_sorter_idx(matlab_sorter_idx, matlab_sorter_plane),'All_planes_rough')
                    #             #%
                                
                    if full_raster_pyhton_cell_idx_list:
                        all_index_info=[ analysis.convert_full_planes_idx_to_single_plane_final_indx(full_raster_pyhton_cell_idx,plane)  for full_raster_pyhton_cell_idx in full_raster_pyhton_cell_idx_list]
                        
                    # get tomato identity    
                    cell_identity=analysis.indetify_full_rater_idx_cell_identity(full_raster_cell_python_idx, plane)
                    if all_index_info:
                        all_index_info=[(cell, analysis.indetify_full_rater_idx_cell_identity(cell[2], plane))for cell in all_index_info]
                        
                        
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm, drifting_options)
                        
    
                #%% explore analysis object

                    
                    # check caiman rasters
                    for dataset in analysis.calcium_datasets.values():
                        dataset.most_updated_caiman.CaimanResults_object.plot_final_rasters()
                
                
    
                    analysis.plot_all_planes_by_cell_type()
                    analysis.full_data_list
                    analysis.reload_other_full_data(-1)
                    imagingdata=full_data['imaging_data']
                    imagingdata['Plane1']['CellNumber']
                    analysis.pyr_int_ids_and_indexes
                    analysis.pyr_int_identification
                    # test=analysis.caiman_results['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000_Plane1_Green']
                    # A=test.caiman_object.cnm_object.estimates.A
                    
                    # dims=(256,256)
                    # i=180
                    # M = A > 0
                    # plt.figure(); plt.imshow(np.reshape(A[:,i-1].toarray(), dims, order='F'))
                    # plt.figure(); plt.imshow(np.reshape(M[:,i-1].toarray(), dims, order='F'))
                    
                    # to transfer to matlab
                    # stim_table_matlab=analysis.stimulus_table['drifting_gratings'].copy()
                    # stim_table_matlab[['start', 'end']]=stim_table_matlab[['start', 'end']]+1
                    # OutData = {} 
                    
                    
                    # output={k:i+1 for k,i in analysis.full_data['visstim_info']['Paradigm_Indexes'].items()}
    
                    # # convert DF to dictionary before loading to your dictionary
                    # OutData['Obj'] = stim_table_matlab.to_dict('list')
                    
                    # sio.savemat(r'C:\Users\sp3660\Desktop\Paradigm.mat',OutData)
                    # sio.savemat(r'C:\Users\sp3660\Desktop\Paradigm.mat',output)
                                    
                    plane='Plane1'
                    cell=1
                    # trace_type='mcmc_binary'
                    trace_type='mcmc_smoothed'

                    correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                    analysis.plot_orientation(correctedcell,trace_type,plane)
                    # analysis.plot_blank_sweeps(cell,trace_type,plane)
                    # analysis.plot_directions(cell,trace_type,plane)
                    for cell in range(14,15):
                        correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                        analysis.plot_orientation(correctedcell,trace_type,plane)
                    
                    # ori=45
                    # stimsequence=np.arange()
                    # test=analysis.stimulus_table['drifting_gratings'][analysis.stimulus_table['drifting_gratings'].orientation==ori]
        
                    
                    # fig,ax=plt.subplots(1)
                    # test=[]
                    # for k in range(1,9):
                    #     ax.vlines(test[k-1], ymin=k+0, ymax=k+1, color=color[k])
        
                #%% TO DO manual GETING THE VISSTIM INDEXES

                elif 'do_visualstim_indexing'  in self.selected_things_to_do:
                    
                    self.destroy()
                    return
                    analysis.signals_object.process_all_signals()
                    analysis.create_full_data_container()
                    analysis.create_stim_table()
                #%% crf preparation
                elif 'crf_prep' in self.selected_things_to_do:
                    self.destroy()
                    return
                    analysis.signals_object.create_full_recording_grating_binary_matrix()
                    
                    grating_Binary_Maytrix_downsampled=np.vstack([analysis.resample(analysis.signals_object.full_stimuli_binary_matrix[srtim], 
                                                                           factor=analysis.milisecond_period, kind='linear').squeeze() for srtim in range (analysis.signals_object.full_stimuli_binary_matrix.shape[0])]),
                    udf=grating_Binary_Maytrix_downsampled[0]
                    
                    twoherzonly=udf[8:16,:]
                    twoherzonly.sum(1)
                    twoherzonly[3,(twoherzonly[3]>0) & (twoherzonly[3]<1)]=0
                    

                    
                    #getcorrdinates and centers
                    all_planes_center_of_mass=np.empty((0,3))
                    for k,v in analysis.caiman_results.items():
                        plane_id=int(k[k.find('_Plane')+6])
                        planecolumn=np.full((v.accepted_center_of_mass.shape[0],1), plane_id)
                        fullplane_array=np.hstack(( v.accepted_center_of_mass , planecolumn))
                        all_planes_center_of_mass=np.vstack((all_planes_center_of_mass,fullplane_array))
                        
     
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm, drifting_options)
                    
                    sio.savemat(os.path.join(analysis.data_paths['CRFs_runs_path'],'data.mat'),{'data':activity_arrays[1].T})
                    sio.savemat(os.path.join(analysis.data_paths['CRFs_runs_path'],'UDF.mat'), {'UDF':twoherzonly.T})
                    sio.savemat(os.path.join(analysis.data_paths['CRFs_runs_path'],'ROIs.mat'),{'ROIs':all_planes_center_of_mass})




                 #%% jesusanalysis new

                elif 'do_jesus_ensemble'  in self.selected_things_to_do:
                    
                    self.destroy()
                    return
                

                    #%% RUN NEW JESUS ANALYSIS ALL POSIBLE SLICES COMBINATIONS
                    print( final_binary_trace_types)
                    print(planes)
                    print(paradigms)
                    print(selected_cells_options)
                    
                    for paradigm in paradigms[2:]:
                        for plane in planes[0:2]:
                            for trace_type in final_binary_trace_types[1:]:
                                for selected_cells in selected_cells_options[:3]:
                                    activity_arrays= analysis.get_raster_with_selections(trace_type, plane, selected_cells, paradigm=paradigm) 
                                    analysis.run_jesus_analysis(activity_arrays)
                                    
                                    
                    paradigm=paradigms[2]
                    plane = planes[0]
                    for trace_type in final_binary_trace_types[0:]:
                        for selected_cells in selected_cells_options[:3]:
                            activity_arrays= analysis.get_raster_with_selections(trace_type, plane, selected_cells, paradigm=paradigm) 
                            analysis.run_jesus_analysis(activity_arrays)
                        
                    #%% CHECKING JESUS RESULTS
                    
                    def intersection(lst1, lst2):
                        return list(set(lst1) & set(lst2))

                    analysis.check_all_jesus_results()
                    pprint(analysis.jesus_results_list)
                    pprint(final_binary_trace_types)
                    pprint(planes)
                    pprint(paradigms)
                    pprint(selected_cells_options)
            
                    #results by cell type
                    pyr_results=[i for i in  analysis.jesus_results_list if (('Pyr' in i) and ('.pkl' in i))]
                    int_results=[i for i in  analysis.jesus_results_list if (('_Interneurons_' in i) and ('.pkl' in i))]
                    all_cells_results=[i for i in  analysis.jesus_results_list if (('_All_jes' in i) and ('.pkl' in i))]

                    #results by paradigm
                    full_movie_results=[i for i in  analysis.jesus_results_list if (('_Full' in i) and ('.pkl' in i))]
                    drift_grat_results=[i for i in  analysis.jesus_results_list if (('Drifting' in i) and ('.pkl' in i))]
                    movie1_results=[i for i in  analysis.jesus_results_list if (('Movie1' in i) and ('.pkl' in i))]
                    movie2_results=[i for i in  analysis.jesus_results_list if (('Movie3' in i) and ('.pkl' in i))]
                    spont_results=[i for i in  analysis.jesus_results_list if (('Spont' in i) and ('.pkl' in i))]


                    #results by trace type
                    mcmc_results=[i for i in  analysis.jesus_results_list if (('mcmc' in i) and ('.pkl' in i))]
                    dfdtresults=[i for i in  analysis.jesus_results_list if (('dfdt' in i) and ('.pkl' in i))]

     
                    pyr_grat=intersection(pyr_results, drift_grat_results)
                    int_grat=intersection(int_results, drift_grat_results)
                    all_cells_grat=intersection(all_cells_results, drift_grat_results)

                    
                    #%% LOAdING JESUS RESULTS 
                    # results get loaded to jesus runs to compare betwen runs
                    analysis.unload_all_runs()
                    analysis.load_jesus_results(mcmc_results[0])
                    # analysis.load_jesus_results(pyr_grat[0])
                    # analysis.load_jesus_results(int_grat[0])
                    # analysis.load_jesus_results(all_cells_grat[0])
                    #%% ANALYSIS SINGLE RESULT RUN
                    analysis.unload_all_runs()
                    analysis.load_jesus_results(mcmc_results[0])
                    analysis.jesus_runs
                    jesusres_object=analysis.jesus_runs[list(analysis.jesus_runs.keys())[0]]
                    jesusres_object.load_analysis_from_file()
                    jesusanalysis=jesusres_object.analysis
                    jesusoptions=jesusres_object.input_options
  
                    # %UMMARY PLOTTING
                    #%
                    jesusres_object.plot_raster()
                    jesusres_object.plot_sorted_rasters()
                    jesusres_object.plot_networks()
                    #% THIS ONE IS SLOW WAIT OFR IT
                    jesusres_object.plot_vector_clustering()
                    
                    # ENSEMBLE PLOTTING TO WORK ON
                    act=jesusanalysis['Ensembles']['ActivationSequence']
                    
         
                    
         
            #%% COMPARE CELL TYPE RUNS  LOAD RUNS
                    plt.close('all')

                    analysis.unload_all_runs()
                    
                    analysis.load_jesus_results(mcmc_results[0])
                    analysis.load_jesus_results(mcmc_results[1])
                    analysis.load_jesus_results(mcmc_results[2])
                    
                    pyr=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['pyr'][1]).flatten()
                    inter=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['int'][1]).flatten()                   
                    cell_subtype_runs={}
                    for run, run_object in  analysis.jesus_runs.items():
                       
                        ensemble_cell_identity={}
                        jesusanalysis=run_object.analysis
                        run_object.plot_raster()
                        run_object.plot_sorted_rasters()
                        run_object.plot_networks()
                        #% THIS ONE IS SLOW WAIT OFR IT
                        run_object.plot_vector_clustering()
                        
                        
                        
                        # dpi = 100
                        # fig = plt.figure(figsize=(16,9),dpi=dpi)
                        # plt.imshow(cell_subtype_runs[list(cell_subtype_runs.keys())[2]][2], cmap='binary', aspect='auto')
                        vectors=jesusanalysis['Ensembles']['Activity']
                        
                        esnemblesequence=jesusanalysis['Ensembles']['ActivationSequence']

                        
                        
                        for ensemble, cells in  enumerate(jesusanalysis['Ensembles']['EnsembleNeurons']):
                       
                            if 'Interneurons_jesus' in run:
                                cells=[inter[cell] for cell in cells ]
                            if 'Pyramidal_jesus' in run:
                                cells=[pyr[cell] for cell in cells ]

                            ensemble_cell_identity['Ensemble: '+str(ensemble+1)]={'Pyramidals':[cell for cell in cells if cell in pyr],
                                                                                  'Interneurons':[cell for cell in cells if cell in inter] ,
                                                                                  
                                }
                            
                        cell_subtype_runs[run]=[run_object,ensemble_cell_identity, vectors]
                        
                    # an=cell_subtype_runs[list(cell_subtype_runs.keys())[0]][0].analysis
                    # cell_subtype_runs[list(cell_subtype_runs.keys())[0]][0].analysis['Ensembles']['EnsembleNeurons']
                    
                    #%% COMPARE ENSMEBLE SIMILARITES
                    plt.close('all')

                    combinedraster=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]][2].astype('float')
                    pyramidalraster= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]][2].astype('float')
                    interneuronraster= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]][2].astype('float')
                        
                    adjacency_pyr=np.zeros((combinedraster.shape[0],pyramidalraster.shape[0] ))
                    adjacency_inter=np.zeros((combinedraster.shape[0],interneuronraster.shape[0] ))


                    for i in range(combinedraster.shape[0]):
                        combined_plus_pyr=np.vstack((combinedraster[i,:],pyramidalraster))
                        combined_plus_inter=np.vstack((combinedraster[i,:],interneuronraster))

                        adjacency_full_pyr=squareform(1-pdist(combined_plus_pyr,'cosine'))
                        adjacency_full_inter=squareform(1-pdist(combined_plus_inter,'cosine'))

                        adjacency_full_pyr[np.isnan(adjacency_full_pyr)]=0 
                        adjacency_full_inter[np.isnan(adjacency_full_inter)]=0 

                        adjacency_inter[i,:]=adjacency_full_inter[0,1:]
                        adjacency_pyr[i,:]=adjacency_full_pyr[0,1:]
                    
                    dpi = 100
                    
                    fig,ax=plt.subplots(1,2, figsize=(20,9))
                    pyr_ad=ax[0].imshow(zscore(adjacency_pyr), cmap='jet', aspect='auto')
                    int_ad=ax[1].imshow(zscore(adjacency_inter), cmap='jet', aspect='auto')
                    fig.colorbar(int_ad, ax=ax[1], location='right', anchor=(0, 0.3), shrink=0.7)
                    fig.suptitle('Z-Scored Ensemble Cosine Similarity')
                    ax[0].set_ylabel('Combined Ensembles')
                    ax[1].set_ylabel('Combined Ensembles')
                    ax[0].set_xlabel('Pyramidal Ensembles')
                    ax[1].set_xlabel('Interneuron Ensembles')
                    plt.tight_layout()

                    
                    fig,ax=plt.subplots(1,2, figsize=(20,9))
                    pyr_ad=ax[0].imshow(adjacency_pyr, cmap='jet', aspect='auto')
                    int_ad=ax[1].imshow(adjacency_inter, cmap='jet', aspect='auto')
                    fig.colorbar(int_ad, ax=ax[1], location='right', anchor=(0, 0.3), shrink=0.7)
                    fig.suptitle('Ensemble Cosine Similarity')
                    ax[0].set_ylabel('Combined Ensembles')
                    ax[1].set_ylabel('Combined Ensembles')
                    ax[0].set_xlabel('Pyramidal Ensembles')
                    ax[1].set_xlabel('Interneuron Ensembles')
                    plt.tight_layout()

                    filename = os.path.join(os.path.split(run_object.analysis_path)[0],f'{"_".join(run_object.input_options)}_{run_object.timestr}_ensmble_similarity.pdf')
                    run_object.save_multi_image(filename)

                    plt.close('all')

                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]

                    
                    similarities={}
                    for name,ensemble in combined[1].items():
                        print('Combined ' + name)
                        similarities[name]={}
                        similarities[name]['Pyramidals']={}
                        similarities[name]['Interneurons']={}
                        similarities[name]['Sizes']={}
                        similarities[name]['Sizes']['Pyramidals']=len(ensemble['Pyramidals'])
                        similarities[name]['Sizes']['Interneurons']=len(ensemble['Interneurons'])
                        similarities[name]['Sizes']['Total']=len(ensemble['Pyramidals'])+len(ensemble['Interneurons'])


                        for name2, pyramidal_ensemble in pyramidals[1].items():
                           x=intersection(ensemble['Pyramidals'], pyramidal_ensemble['Pyramidals'])
                           intersect=len(x)
                           union=(len(ensemble['Pyramidals'])+len(pyramidal_ensemble['Pyramidals'])-intersect)
                           simil=intersect/union

                            # print('Combined ' +name+': Pyramidal'+name2)
                            # print(len(x))
                            # print(len(ensemble['Pyramidals']))
                            # print(100*len(x)/len(ensemble['Pyramidals']))
                            # print(len(x))
                            # print(len(pyramidal_ensemble['Pyramidals']))
                            # print(100*len(x)/len(pyramidal_ensemble['Pyramidals']))
                           sizes1={'Pyramidals':len(pyramidal_ensemble['Pyramidals']),
                                  'Interneurons':0,
                                  'Total':len(pyramidal_ensemble['Pyramidals'])}
                           similarities[name]['Pyramidals'][name2]=[x,simil, sizes1]

                           
                        for name3, interneurons_ensemble in interneurons[1].items():

                           x=intersection(ensemble['Interneurons'], interneurons_ensemble['Interneurons'])
                           intersect=len(x)
                           union=(len(ensemble['Interneurons'])+len(interneurons_ensemble['Interneurons'])-intersect)
                           simil=intersect/union
                           sizes2={'Pyramidals':0,
                                  'Interneurons':len(interneurons_ensemble['Interneurons']),
                                  'Total':len(interneurons_ensemble['Interneurons'])}


                           similarities[name]['Interneurons'][name3]=[x,simil, sizes2]



                    for test_ensem, val in similarities.items():
    
                        intersimilaritesenemble1=[ensemble[1] for ensemble in val['Interneurons'].values()]
                        pyrsimilaritesenemble1=[ensemble[1] for ensemble in val['Pyramidals'].values()]
                        pyrensesize=[ensemble[2]['Pyramidals'] for ensemble in val['Pyramidals'].values()]
                        intensesize=[ensemble[2]['Interneurons'] for ensemble in val['Interneurons'].values()]
                        fullsizepyr=val['Sizes']['Pyramidals']
                        fullsizeint=val['Sizes']['Interneurons']
                        fullsizetot=val['Sizes']['Total']
                        test_ensem_l=['Combined'+ test_ensem]
                        test_ensem_l.extend(list(val['Pyramidals'].keys()))
                        test_ensem_int=['Combined'+ test_ensem]
                        test_ensem_int.extend(list(val['Interneurons'].keys()))

                        twinaxes=[0,0]
                        colors = ['r','k','k','k','k','k','k','k','k','k','k']

                        pyrensesize.insert(0,fullsizepyr)
                        pyrsimilaritesenemble1.insert(0,fullsizepyr/fullsizetot)
                        fig,ax=plt.subplots(1,2, sharey=True)
                        ax[0].bar(test_ensem_l, pyrsimilaritesenemble1, color=colors)
                        twinaxes[0]=ax[0].twinx()
                        twinaxes[0].plot(np.arange(0,len(pyrensesize)), pyrensesize, 'b')
                        
                        intensesize.insert(0,fullsizeint)
                        intersimilaritesenemble1.insert(0,fullsizeint/fullsizetot)
                        ax[1].bar(test_ensem_int, intersimilaritesenemble1, color=colors)
                        twinaxes[1]=ax[1].twinx()
                        twinaxes[1].plot(np.arange(0,len(intensesize)), intensesize,'b')

                        twinaxes[0].set_ylim(0, 30)
                        twinaxes[1].set_ylim(0, 35)
                        ax[0].set_title('Pyramidal Ensembles')
                        ax[0].set_ylabel('Similarity')

                        ax[1].set_title('Interneuron Ensembles')


                        ax[0].set_ylim(0, 1.2)
                        ax[0].set_xticklabels(test_ensem_l, rotation = 90)
                        ax[1].set_xticklabels(test_ensem_l, rotation =90)
                        twinaxes[1].set_ylabel('Cell Number')
                        fig.suptitle(test_ensem_l[0])
                        plt.tight_layout()
                        
                    filename = os.path.join(os.path.split(run_object.analysis_path)[0],f'{"_".join(run_object.input_options)}_{run_object.timestr}_ensemble_by_ensemble.pdf')
                    run_object.save_multi_image(filename)

                    plt.close('all')
                    

 #%% COMPARISON OF ESNEMBLE BETWEN THRE CELL YPES RUNS

                    indexes=((analysis.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_first'],
                    analysis.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_last']),
                    (analysis.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_first'],
                    analysis.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_last']),
                    (analysis.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_first'],
                    analysis.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_last'])) 
                    
                    indexes=((analysis.full_data['visstim_info']['Paradigm_Indexes']['first_movie_set_first'],analysis.full_data['visstim_info']['Paradigm_Indexes']['first_movie_set_last']),
                    (analysis.full_data['visstim_info']['Paradigm_Indexes']['short_movie_set_first'],analysis.full_data['visstim_info']['Paradigm_Indexes']['short_movie_set_last']),
                    (analysis.full_data['visstim_info']['Paradigm_Indexes']['second_movie_set_first'],analysis.full_data['visstim_info']['Paradigm_Indexes']['second_movie_set_last']))
                    
                    
                    
                    trace_type='mcmc_binary'
                    trace_type='mcmc_smoothed'
                    plane='All_planes_rough'
                    ensmeble='Ensemble: 2'
                    
                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]
                    
                    
                    celtyp='Pyramidals'
                    ense=pyramidals[1][ensmeble][celtyp]
                    
                    celtyp='Interneurons'
                    ense=interneurons[1][ensmeble][celtyp]
                    
                    cells=np.array(ense)
                    
                    
                    
                    celtyp='All'
                    cells=np.concatenate([np.array(ensemble_cell_identity[ensmeble]['Pyramidals']).astype('int'), np.array(ensemble_cell_identity[ensmeble]['Interneurons']).astype('int')])

                   

                    test=analysis.full_data['imaging_data'][plane]['Traces'][trace_type]
                    paradigm_sliced_raster= analysis.slice_matrix_by_paradigm_indexes(test, indexes)
                    paradigm_sliced_speed= analysis.slice_matrix_by_paradigm_indexes(np.expand_dims(analysis.full_data['voltage_traces']['Speed'],axis=0), indexes).squeeze()

                   
                    pixel_per_bar = 4
                    dpi = 100
                    # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
                    fig , ax= plt.subplots(2,figsize=(16,9), dpi=dpi)
                    ax[0].imshow(paradigm_sliced_raster[cells,:], cmap='binary', aspect='auto',
                        interpolation='nearest', norm=mpl.colors.Normalize(0, 0.1))
                    ax[0].set_xlabel('Time (s)')
                    ax[1].plot(paradigm_sliced_speed)
                    ax[1].margins(x=0)
                    fig.supylabel('Cell')
                    fig.suptitle(f'{celtyp}_{ensmeble}')
                    

                    
                    
                    
                    
                    ensemble=list(similarities.keys())[1]   
                    cells=copy.copy(combined[1][ensemble]['Pyramidals'])
                    cells.extend(combined[1][ensemble]['Interneurons'])
                    analysis.plot_orientation(352,trace_type,plane)

                    # for cell in cells:
                    #     #%%
                    #     correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                    #     #%%
                    #     analysis.plot_orientation(correctedcell,trace_type,plane)
                    # #%%
                    
                    
                    all_mena_act=[]
                    for ensemble in similarities['Ensemble: 1']['Pyramidals'].keys():
                        cells=copy.copy(pyramidals[1][ensemble]['Pyramidals'])
                        meanacti=[]
                        for cell in cells:
                            correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                            s2,_=analysis.plot_orientation(correctedcell,trace_type,plane)
                            meanacti.append((s2))
    
    
                        meanoriactensemble=np.zeros((4,1))
                        test=list(zip(*meanacti))
                        meani=[]
                        for j,i in enumerate(test):
                            meanoriactensemble[j]=(np.mean(i))
                        all_mena_act.append(meanoriactensemble)
                        
                    fig,ax=plt.subplots(1)
                    for i,ensembl in enumerate(all_mena_act):
                        ax.plot([0,45,90,135],ensembl, label='Ensemble: {}'.format(i+1))
                        ax.set_xticks([0,45,90,135])        # set xtick values

                    ax.legend(loc=2)
                    
                    
                    all_mena_act=[]
                    for ensemble in similarities['Ensemble: 1']['Interneurons'].keys():
                        cells=copy.copy(interneurons[1][ensemble]['Interneurons'])
                        meanacti=[]
                        for cell in cells:
                            correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                            s2,_=analysis.plot_orientation(correctedcell,trace_type,plane)
                            meanacti.append((s2))
    
    
                        meanoriactensemble=np.zeros((4,1))
                        test=list(zip(*meanacti))
                        meani=[]
                        for j,i in enumerate(test):
                            meanoriactensemble[j]=(np.mean(i))
                        all_mena_act.append(meanoriactensemble)
                        
                    fig,ax=plt.subplots(1)
                    for i,ensembl in enumerate(all_mena_act):
                        ax.plot([0,45,90,135],ensembl, label='Ensemble: {}'.format(i+1))
                        ax.set_xticks([0,45,90,135])        # set xtick values

                    ax.legend(loc=2)
                    
                    
                    
                    
                    all_mena_act=[]
                    for ensemble in ensemble_cell_identity.keys():
                        
                        cells=np.append(np.array(ensemble_cell_identity[ensemble]['Pyramidals']),
                                        np.array(ensemble_cell_identity[ensemble]['Interneurons']))
                        
                        meanacti=[]
                        for cell in cells:
                            correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                            s2,_=analysis.plot_orientation(correctedcell,trace_type,plane)
                            meanacti.append((s2))
    
    
                        meanoriactensemble=np.zeros((4,1))
                        test=list(zip(*meanacti))
                        meani=[]
                        for j,i in enumerate(test):
                            meanoriactensemble[j]=(np.mean(i))
                        all_mena_act.append(meanoriactensemble)
                        
                    fig,ax=plt.subplots(1)
                    for i,ensembl in enumerate(all_mena_act):
                        ax.plot([0,45,90,135],ensembl, label='Ensemble: {}'.format(i+1))
                        ax.set_xticks([0,45,90,135])        # set xtick values

                    ax.legend(loc=2)
                    
            
                    # test=pyramidals[0]
                    # selected_ensemble=1
                    # promiscous_cells={}
                    
                    
                    # for cell in jesusanalysis['Ensembles']['EnsembleNeurons'][selected_ensemble-1]:
                    #     promiscous_cells[str(cell)]=[]
                    #     for ensemble, ensemble_cells in enumerate(jesusanalysis['Ensembles']['EnsembleNeurons']):
                    #         if ensemble!=selected_ensemble+1:
                    #             if cell in ensemble_cells:
                    #                promiscous_cells[str(cell)].append( ensemble)
                                   
                                   
                    # promiscous_cells_pyrs={cell: promiscous_cells[str(cell)] for cell in ensemble_cell_identity['Ensemble: '+str(selected_ensemble)]['Pyramidals'] }
                    # promiscous_cells_interneurons={cell: promiscous_cells[str(cell)] for cell in ensemble_cell_identity['Ensemble: '+str(selected_ensemble)]['Interneurons'] }

                    
                    
                    
                    # plt.close('All')
                    # plane='Plane1'
                    # trace_type='dfdt_smoothed'
                    # preframes=16
                    # stim=33
                    # postframes=16


                    # for cell in ensemble_cell_identity['Ensemble: '+str(selected_ensemble)]['Pyramidals']:
                    #     matlabcell=analysis.full_data['imaging_data'][plane]['CellIds'][cell]
                    #     if  cell in    pyr:
                    #         celltype='Pyramidal Cell'
                    #     elif  cell in    inter:
                    #         celltype='Interneuron'
                    #     print(plane+'\nMatlab cell: '+str( matlabcell)+'\nPython cell :'+str(cell)+'\n' + celltype)
                    #     print(promiscous_cells_interneurons)
                    #     plot_orientation(cell, trace_type, plane)  
                        
                  

#%% allen analysys
                if 'do_tuning_allen'  in self.selected_things_to_do:
                    analysis.load_allen_analysis()
                    self.destroy()
                    return
                    analysis.allen_analysis.set_up_drifting_gratings_parameters()
                    analysis.allen_analysis.do_AllenA_analysis(plane, matrix)
                    # peak_info=analysis.allen_analysis.peak
                    # reponse_info=analysis.allen_analysis.response 
                
                    print( trace_types)
                    print(planes)
                    print(paradigms)
                    print(selected_cells_options)
                    plane=planes[0]
                    matrix=trace_types[3]
                    cell_selecton_index=1
                    cell_type=selected_cells_options[cell_selecton_index]

                    pyr=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['pyr'][1]).flatten()
                    inter=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['int'][1]).flatten()     
                    all_cells=np.concatenate((pyr, inter))
                    cells=(all_cells,pyr,inter)


                    
                    for matrix in trace_types:
                        for i, cell_type in enumerate( selected_cells_options[:-1]):
                            analysis.allen_analysis.do_AllenA_analysis(plane, matrix)
                            selectedcells=cells[i]
                            params=(matrix,plane,cell_type)
                            analysis.do_PCA(analysis.allen_analysis.mean_sweep_response.iloc[:,selectedcells], analysis.allen_analysis.sweep_response.iloc[:,selectedcells], analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'], params)

                    
                             
#%% yuriy ensembles

                if 'do_yuriy_ensembles'  in self.selected_things_to_do:
                    analysis.load_yuriy_analysis()
                    self.destroy()
                    return
                    
      
    if __name__ == "__main__":
        
        
        pass    