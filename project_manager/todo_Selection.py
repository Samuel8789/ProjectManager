# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:57:50 2022

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter
import os
import copy
import pickle
import glob
from pprint import pprint
from pathlib import Path
from sys import platform
import socket
import urllib3
import warnings
warnings.filterwarnings("ignore")

from IPython.display import HTML
import sys
import caiman as cm

import numpy as np
import pandas as pd

import scipy as spy
import scipy.io as sio
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial.distance import squareform, pdist
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind, zscore, mode, norm


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation 
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

# sys.path.insert(0, r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_analysis')
sys.path.insert(0, os.path.join(os.path.expanduser('~'),r'Documents/Github/LabNY/ny_lab/data_analysis'))

from jesusEnsemblesResults import JesusEnsemblesResults



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
        #Data processing 7:13
        'testing_datamanagin',
        'testing_image_processing',
        'testing_image_analysis',
        'deep_caiman',
        'dataset_focus',
        'voltage_vis_stim_analysis',
        #Analysis 13:
        'do_data_analysis_from_non_database',
        'do_data_analysis_from_database',
        'explore_analysis',
        'do_visualstim_indexing',
        'do_jesus_ensemble',
        'do_tuning_allen',
        'do_yuriy_ensembles',
        #analysis no dtamanaging
        'do_cloud_results_only'
        ]
        
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

            self.to_return=[lab,[], [],[],[],[], [],[],[],[], [],[],[]]

#%% ALLEN BO
        elif self.things_to_do[6] in self.selected_things_to_do:
            
            allen=self.projectManager.initialize_a_project('AllenBrainObservatory',self.gui)  
            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            self.to_return=[lab, allen,[],[],[],[], [],[],[],[], [],[],[]]

            self.destroy()
            return
            allen.set_up_monitor( screen_size=[750,1280], lenght=37.7)
            #%%
            allen.get_visual_templates()
            #%%
            # allen.get_gratings()
            # allen.get_drifting_gratings()
            allen.get_selection_options()
            
            
            '''
            MAin selection here
            '''
            area='VISp'
            line='Vglut1'
            # line='SST'
            # line='Vip'
            # line='PV'

            depth=175
            stim='drifting_gratings'
            '''
            MAin selection here
            '''
            
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
            three_exp_container_directory=os.path.join(allen.main_directory,'Containers',str(exps_by_fov['experiment_container_id'][0]))
            if not os.path.isdir(three_exp_container_directory):
                os.mkdir(three_exp_container_directory)
                
            
            
            for i in exps_list_by_fov:
                all_exp_fov.append(allen.download_single_imaging_session_nwb(i['id']))
                
                if not os.path.isdir(os.path.join(three_exp_container_directory,str(i['id']))):
                    os.mkdir(os.path.join(three_exp_container_directory,str(i['id'])))
                
                

            filename='Allen_{}_{}_{}_{}'.format(line, area, depth, selected_container_id )
            for i in range(3):
                print(all_exp_fov[i][-1])
            
            selection=1
            data_set=all_exp_fov[selection][0]
            spikes=all_exp_fov[selection][1]
            plt.imshow(spikes, cmap='binary', aspect='auto', vmax=0.01)

          
            #%% allen jesus analysis
            from ny_lab.data_analysis.resultsAnalysis import ResultsAnalysis
            allen_results_analysis=ResultsAnalysis(allen_BO_tuple=(allen,data_set,spikes ),new_full_data=True)
            # allen_results_analysis=ResultsAnalysis(allen_BO_tuple=(allen,data_set,spikes ))

            
            #decide an pprepare array for jesus analysis
            #%%
            from numpy.random import default_rng
            seed=0
            rng = default_rng(seed)
            cellsubsample=40

            indexed_cells = rng.choice(allen_results_analysis.full_data['imaging_data']['Plane1']['CellNumber'], size=cellsubsample, replace=False)
            indexed_cells.sort()
            allen_results_analysis.binarization(0.08)
            
            trace_type='binarized'
            paradigm='Drifting_Gratings' 
            # selected_cells='All'
            selected_cells=indexed_cells.tolist()

            activity_arrays= allen_results_analysis.get_raster_with_selections(trace_type,'Plane1',selected_cells, paradigm)
            plt.imshow(activity_arrays[3], cmap='binary', aspect='auto', vmax=0.01)

            #run new jesus analys
            allen_results_analysis.run_jesus_analysis(activity_arrays)
            allen_results_analysis.save_activity_array_to_matlab(activity_arrays)           
   
            #selec and load a saved jesus analysys
            allen_results_analysis.check_all_jesus_results()
            allen_results_analysis.unload_all_runs()
            allen_results_analysis.jesus_runs
            pprint(allen_results_analysis.sorted_jesus_results)
            #%% selec here change all time
            allen_results_analysis.load_jesus_results(allen_results_analysis.sorted_jesus_results['selected_cells_grat'][1])
            # allen_results_analysis.load_jesus_results(allen_results_analysis.sorted_jesus_results['all_cells_grat_binarized'][0])
            pprint(allen_results_analysis.jesus_runs)
            jesusres_object=allen_results_analysis.jesus_runs[list(allen_results_analysis.jesus_runs.keys())[0]]
            jesusres_object.load_analysis_from_file()
            jesusanalysis=jesusres_object.analysis
            jesusoptions=jesusres_object.input_options
           #%plotting jesus results
            jesusres_object.plot_raster()
            jesusres_object.plot_sorted_rasters()
            jesusres_object.plot_networks()  
            #% THIS ONE IS SLOW WAIT OFR IT
            jesusres_object.plot_vector_clustering()
            
               
            #%%

            
            ts,traces=data_set.get_demixed_traces()
            
            selected_cell=3
            f,axs=plt.subplots(2,sharex=True)
            axs[0].plot(traces[selected_cell,:])
            axs[1].plot(spikes[selected_cell,:])
            
     
            tt=data_set.get_roi_mask()
                
         #%% this is for plotting the cells
            import numpy as np
            from matplotlib import pyplot as plt
            from matplotlib.widgets import Slider
            import numpy.ma as ma
            from matplotlib.colors import Normalize
            import matplotlib.cm as cmm
            from matplotlib.patches import Rectangle
            # plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            
            ts,traces=data_set.get_corrected_fluorescence_traces()
            tt=data_set.get_roi_mask()
            allmasks=[i.get_mask_plane() for i in tt]

            allboundaries=np.zeros(list(tt[0].get_mask_plane().shape)+ [len(tt)])
            mockmovie=np.zeros([20,20,traces.shape[1]])
            noise = np.random.normal(0, .1, mockmovie.shape)
            mockmovie=mockmovie+noise
            
            

            for j,cell in enumerate(tt):
                zz=cell.get_mask_plane()
                x=np.diff(zz)[1:-1,1:]
                y=np.diff(zz,axis=0)[1:,1:-1]
                z=x+y
                zzz=np.where(z!=0)
                xcoords=zzz[0]+1
                ycoords=zzz[1]+1
                canv=np.zeros(tt[0].get_mask_plane().shape)
                for i in  np.c_[xcoords, ycoords]:
                    canv[i[0],i[1]]=1
                    canv[np.where(canv ==0)] = np.nan
                allboundaries[:,:,j]=canv
                
            allboundariesprojecy=np.nansum(allboundaries, axis=2)
            allboundariesprojecy[np.where(allboundariesprojecy ==0)] = np.nan


            fig, ax = plt.subplot_mosaic(
                [['a)', 'b)','d)','f'], 
                 ['c)', 'c)','c)','c)'],
                 ['e','e','e','e']],)
            my_cmap = cmm.jet
            my_cmap.set_under('k', alpha=0)
           
  
    
            cellid=0
            from scipy import ndimage
           
            def ax_update(ax):
                ax.set_autoscale_on(False)  # Otherwise, infinite loop
                # Get the number of points from the number of pixels in the window
                width, height = \
                    np.round(ax.patch.get_window_extent().size).astype(int)
                # Get the range for the new area
                vl = ax.viewLim
                extent = vl.x0, vl.x1, vl.y0, vl.y1
                print(vl)

                # Update the image object with our new data and extent
                ax.figure.canvas.draw_idle()


            def plot_selected_cell(cellid):
                centers=ndimage.measurements.center_of_mass(allmasks[cellid])
                squaresize=20
                xcent=np.rint(centers[0]).astype(np.uint16)
                ycent=np.rint(centers[1]).astype(np.uint16)
                xorigin=xcent-squaresize
                xend=xcent+squaresize
                yorigin=ycent-squaresize
                yend=ycent+squaresize
                if xorigin<0:
                    xorigin=0
                if yorigin<0:
                    yorigin=0
                if xend>allmasks[0].shape[0]:
                    xend=allmasks[0].shape[0]
                if yend>allmasks[0].shape[1]:
                    yend=allmasks[0].shape[1]

                cellfocused=data_set.get_max_projection()[xorigin:xend,yorigin:yend]
                ax['d)'].imshow(cellfocused)
                ax['d)'].imshow(allboundaries[xorigin:xend,yorigin:yend,cellid],cmap='binary')


            def update(val):
               ax['b)'].clear()
               ax['c)'].clear()
               ax['e'].clear()


               ax['b)'].imshow(data_set.get_max_projection())

               ax['b)'].imshow(allboundaries[:,:,slider.val],cmap='binary', interpolation='none',)
               ax['c)'].plot(ts,traces[slider.val,:])
               ax['e'].plot(ts,traces[slider.val,:])

               plot_selected_cell(slider.val)
               
               
               rect = UpdatingRect(
                   [0, 0], 0, 0, facecolor='none', edgecolor='red', linewidth=3.0)
               rect.set_bounds(* ax['e'].viewLim.bounds)
               
               ax['c)'].add_patch(rect)
                                                  
               ax['e'].callbacks.connect('xlim_changed', rect)
               ax['e'].callbacks.connect('ylim_changed', rect)
               
               
               
               fig.canvas.draw_idle()
               
            def mouse_event(event):
                selectedaxes=event.inaxes
                 
                if selectedaxes and not selectedaxes.get_subplotspec().is_last_row() and not selectedaxes.get_subplotspec().is_last_col() :
                   print('x: {} and y: {}'.format(event.xdata, event.ydata))
                   xco=np.rint(event.xdata).astype(np.uint16)
                   yco=np.rint(event.ydata).astype(np.uint16)
                   
                   selectedmasks=[i for i,mask in enumerate(allmasks) if mask[yco,xco]==1]
                   if selectedmasks:
                       slider.set_val(selectedmasks[0])
                       
                   
            def on_press(event):
                # sys.stdout.flush()
                print(event.key)
                if event.key == 'right':
                    slider.set_val(slider.val+1)
                if event.key == 'left':
                    slider.set_val(slider.val-1)
                fig.canvas.draw_idle()
                
            
            
            class UpdatingRect(Rectangle):
               def __call__(self, ax):
                self.set_bounds(*ax.viewLim.bounds)
                ax.figure.canvas.draw_idle()
                
                xleft=np.rint(ax.viewLim.bounds[0]).astype(np.uint16)
                xright=np.rint(ax.viewLim.bounds[1]).astype(np.uint16)
                sliderleftindex=np.argmin(np.abs(ts-xleft))
                sliderrightindex=np.argmin(np.abs(ts-xright))

                
                timeslider_new_range(sliderleftindex, sliderrightindex)
                
                
            def replot_frame(val):
                ax['f'].clear()
                ax['f'].imshow(mockmovie[:,:,timeslider.val])
                line=ax['e'].axvline(x = ts[timeslider.val], color = 'b', label = 'axvline - full height')
                
              
            def timeslider_new_range(xleft, xright):
                timeslider.valmin = xleft
                timeslider.valmax = xright
                timeslider.ax.set_xlim(timeslider.valmin,timeslider.valmax)

                
                    
    
            indexes = np.arange(0, len(tt),1)
            indexes2=np.arange(0, mockmovie.shape[-1],1)

            
            img = ax['a)'].imshow(data_set.get_max_projection())
            im=ax['a)'].imshow(allboundariesprojecy,cmap='binary', interpolation='none',)

            
            img = ax['b)'].imshow(data_set.get_max_projection())
            ax['f'].imshow(mockmovie[:,:,0])



            ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03])
            ax_slider2 = plt.axes([0.8, .65, .15, .02])

            slider = Slider(ax_slider, 'Slide->', 0, len(tt), valstep=indexes, valinit=0)
            timeslider = Slider( ax_slider2, 'Slide->', 0, mockmovie.shape[-1], valstep=indexes2, valinit=0)

            slider.on_changed(update)
            timeslider.on_changed(replot_frame)

            fig.canvas.mpl_connect('key_press_event', on_press)
            fig.canvas.mpl_connect('button_press_event', mouse_event)
            
           
         
   
            plt.show()
         




            #%%
            goodcells=[2,3,6,7,13,15,17,19,21,23,25,26,31]
            
            ttt=data_set.get_roi_mask_array()
            fig, ax = plt.subplots(1)
            ax.imshow(data_set.get_max_projection())
            ax.imshow(np.sum(ttt[goodcells,:,:],axis=0),alpha=0.5)
            drifttable=data_set.get_stimulus_table('drifting_gratings')
            plt.imshow(spikes[goodcells,:], cmap='binary', aspect='auto', vmax=0.01)
            
            
            drifttable['orientation']==0
            

            

#%% DATA PROCESSING
        elif [todo for todo in self.selected_things_to_do if todo in self.things_to_do[7:13]] : 

            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            MouseDat=lab.database
            lab.do_datamanaging()
            datamanaging=lab.datamanaging
            
            self.to_return=[lab, [],MouseDat,datamanaging,[],[], [],[],[],[], [],[],[]]
#%% DATA MANAGING TESTING

            if 'testing_datamanagin' in self.selected_things_to_do:

                self.destroy()
                return
#%%
                #this is to pdate slow and working storage after i chan ge dand finnalt stablized computer
                datamanaging.update_mouse_slow_storages()
#%%
                #hius is for changing the letter for teh data permanet diskj after I tarnsfer everythion to internal ssd
                datamanaging.update_drive_letter_of_permanent_paths()

                # this was done for the disk change i think
                # datamanaging.update_pre_process_slow_data_structure(update=True)
                datamanaging.update_all_imaging_data_paths()
                datamanaging.read_all_data_path_structures()
                # datamanaging.delete_pre_procesed_strucutre_mouse_without_data()
                # datamanaging.read_all_imaging_sessions_from_directories()
                
                datamanaging.read_all_immaging_session_not_in_database()
                
            elif 'voltage_vis_stim_analysis' in self.selected_things_to_do:
                self.destroy()
                return
            
                mousenames=['SPKG']
                mousename=mousenames[0]
    
                mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                allacqs=mouse_object.all_mouse_acquisitions
    
                pprint(list(allacqs.keys()))
                # selectedaqposition = int(input('Choose Aq position.\n'))
                selectedaqposition=6
                
                #% getting acq
                acq=allacqs[list(allacqs.keys())[selectedaqposition]]
                acq.get_all_database_info()
               
#%% DEEP CAIMAN
            elif 'deep_caiman' in self.selected_things_to_do:
                
                self.destroy()
                return
            
            #%%restarting database
                MouseDat.close_database()
                MouseDat.reconnect_database()
                lab.do_datamanaging()
                datamanaging=lab.datamanaging
               #%% 
      
                datamanaging.get_all_deep_caiman_objects()

                datamanaging.all_deep_caiman_objects
                allimagedmice=datamanaging.all_imaged_mice['Code'].unique().tolist()
                allimagedmice.sort()
                
                # tododeepcaiman=['SPHV', 'SPHW','SPHX','SPJB','SPJD','SPKF','SPKH','SPKI','SPKL','SPIG','SPIH']
                tododeepcaiman=['SPGT', 'SPHQ','SPIB','SPIC','SPIL','SPIM','SPIN','SPJF','SPJG','SPJH','SPJI','SPJZ','SPKC','SPKS',	'SPKU'	,'SPKV'	,'SPLE',	'SPLF']
                tododeepcaiman=['SPIL','SPJF']
                tododeepcaiman=['SPOU']
                tododeepcaiman=['SPRE','SPRB']


#%%
                for i in tododeepcaiman:
                    datamanaging.do_deep_caiman_of_mice_datasets([i])
#%% DATASET FOCUS
            elif 'dataset_focus' in self.selected_things_to_do:
                self.destroy()
                return
                
                chand_good_datasets=['SPKS','SPRE','SPRB','SPRN','SPRM','SPRZ']
                int_good_datasets=['SPKG', 'SPOL','SPQZ']
                mousename=chand_good_datasets[4]

                mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                allacqs=mouse_object.all_mouse_acquisitions

                pprint(list(allacqs.keys()))
                # selectedaqposition = int(input('Choose Aq position.\n'))
                selectedaqposition=6
                
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
                mostupdatedcaimanextraccion=dtset.most_updated_caiman
                mostupdatedcaimanextraccion.load_results_object()
                mostupdatedcaimanextraccion.CaimanResults_object.open_caiman_sorter()
                mov=cnm.estimates.A[:,:].toarray()@(cnm.estimates.C+cnm.estimates.YrA)+cnm.estimates.b@cnm.estimates.f
                movob=cm.movie(mov.T.reshape((64415,256,256)))
                #%% OPTIONAL REMOVE OPTO LED FRAMES BEFORE MOTION CORRETCIN AND CIAMAN
                acq.load_all()
                volt=acq.voltage_signal_object
                volt.voltage_signals_dictionary_daq
                volt.voltage_signals_dictionary
                dtset.remove_LED_artifacts()
                dtset.preoptoframes
                dtset.postoptoframes
                cm.load(dtset.shifted_movie_path)
                #%%
                acq.voltage_signal_object.signal_extraction_object()

                
                
                
                
                #%% do filtering and max images after manual motion correction
                dtset.most_updated_caiman.check_motion_corrected_on_acid()
                dtset.read_all_paths()
                dtset.do_initial_kalman(dtset.most_updated_caiman.mc_onacid_path)
                #%%
                dtset.do_summary_images(dtset.kalman_movie_path)
                # loading the excel with opto cell itargeted form sorter
                p=Path(dtset.selected_dataset_mmap_path)
               
                cellfiles=glob.glob(p.parents[0].as_posix()+'\**.xlsx')
                
                df1 = pd.read_excel(cellfiles[0], engine="openpyxl")
                df1.loc[:, "Matlab Sorter Cell"] = df1["Matlab Sorter Cell"].apply(lambda x: x - 1)
                
                acq.get_all_database_info()
                #%%
                acq.load_results_analysis(new_full_data=True) 
                # acq.load_results_analysis(new_full_data=False) 
                
                analysis=acq.analysis_object
                full_data=analysis.full_data
            
                print(acq.aquisition_name)
                
                
                

                #extract opto frames from voltage recordings
                
                
                #%% initial caiman and motion correct
                chand_good_datasets=['SPKS','SPRE','SPRB','SPRM','SPRN','SPRZ','SPST','SPSM','SPSU','SPSX','SPSZ']
                int_good_datasets=['SPKG', 'SPOL','SPQZ','SPRA','SPQW','SPQX']
                mousename=chand_good_datasets[-2]
 
                mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                allacqs=mouse_object.all_mouse_acquisitions
 
                pprint(list(allacqs.keys()))
                # selectedaqposition = int(input('Choose Aq position.\n'))
                selectedaqposition=0
                #rafachandoptodatasetis  selectedaqposition=1
                
                #% getting acq
                acq=allacqs[list (allacqs.keys())[selectedaqposition]]
                acq.get_all_database_info()
               
                acq.load_all()
                volt=acq.voltage_signal_object
                volt.voltage_signals_dictionary_daq
                volt.voltage_signals_dictionary
                database_acq_raw_path=Path(datamanaging.transform_databasepath_tolinux(acq.acquisition_database_info.loc[0, 'AcquisitonRawPath'])).resolve()
                raw_dataset_path= Path(glob.glob(str(database_acq_raw_path)+os.sep+'**')[0])
                
                 
                calciumdatasets={ i:l for i,l in acq.all_datasets.items() if 'Green' in i}
                pprint(calciumdatasets)
                selecteddtsetposition=0         
                #% getting acq
                dtset=calciumdatasets[list(calciumdatasets)[selecteddtsetposition]]
                dtset.open_dataset_directory()

                #%%tesitng galois
                
                # dtset.galois_caiman()
                #%% step by step ciman procesing
                dtset.do_initial_caiman_extraction()
                dtset.do_initial_kalman(dtset.initial_caiman.mc_onacid_path)
                dtset.read_all_paths() #this is to reload the caimanrsults oibject and add the onacid path.
                dtset.do_initial_kalman(dtset.mc_onacid_path)

                dtset.initial_caiman.load_results_object()
                dtset.do_summary_images(dtset.gauss_path)
                #%%
                # dtset.initial_caiman.CaimanResults_object.open_caiman_sorter()
                dtset.read_all_paths() #this is to reload the caimanrsults oibject and add the onacid path.
                dtset.load_dataset()
                #%%
                
                new_param_dict={'nb':1,'movie_slice':np.arange(0,5000)}
                new_param_dict={'nb':1}
                new_param_dict={'epochs':1,'K':200, 'gSig':(5,5)}

                dtset.do_deep_caiman(new_param_dict)
                dtset.read_all_paths() #this is to reload the caimanrsults oibject and add the onacid path.
                dtset.load_dataset()
                dtset.most_updated_caiman.check_caiman_files()
                dtset.most_updated_caiman.caiman_path
                dtset.most_updated_caiman.load_results_object()
                # dtset.most_updated_caiman.CaimanResults_object.open_caiman_sorter()
                
            #%%
                acq.load_metadata_slow_working_directories()
                #%% gsig=4
               

                #%%
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
                    
#%% IMAGE PROCESSING   
            elif 'testing_image_processing' in self.selected_things_to_do:    
                self.destroy()
                return
                sessionnames=['20220223','20220306','20220314','20220331','20220414']
                
                for session_name in sessionnames:
                    

                    prairie_session=datamanaging.all_existing_sessions_database_objects[session_name]
                # this is the celan up and org, this has to be done first
                    prairie_session.process_all_imaged_mice()      
                    
                    
                #%% raw processing
                session_name='20230709'
                prairie_session= datamanaging.all_existing_sessions_not_database_objects[session_name]
                
                # this is the celan up and org, this has to be done first
                prairie_session.process_all_imaged_mice()
                

                
                for mouse_code in prairie_session.session_imaged_mice_codes:
                    prairie_session.datamanagingobject.all_experimetal_mice_objects[mouse_code].raw_imaging_sessions_objects[session_name].raw_session_preprocessing()

                    mouse_object=prairie_session.datamanagingobject.all_experimetal_mice_objects[mouse_code]
                    session_object=mouse_object.raw_imaging_sessions_objects[session_name]
                    mouse_object.get_all_mouse_raw_acquisitions_datasets(mouse_object.raw_imaging_sessions_objects)
                    for dataset in mouse_object.all_raw_mouse_acquisitions_datasets.values():
                        dataset.process_raw_dataset(forcing=True)
                
                #%%
                dataset=list(mouse_object.all_raw_mouse_acquisitions_datasets.values())[2]
                dataset.process_raw_dataset(forcing=True)

                
                    
#%% DATA ANALYSIS                  
        elif [todo for todo in self.selected_things_to_do if todo in self.things_to_do[12:-1]] :   
            
            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            MouseDat=lab.database
            lab.do_datamanaging()
            datamanaging=lab.datamanaging
            
            self.to_return=[lab, [],MouseDat,datamanaging,[],[], [],[],[],[], [],[],[]]
            
    
#%% DATA ANALYSIS NOT FORM DATABASE
            if 'do_data_analysis_from_non_database' in self.selected_things_to_do:
                session_name='20220330'
                mousename='SPKU'
                
                pprint(datamanaging.all_existing_sessions_not_database_objects.keys())
                pprint(datamanaging.all_experimetal_mice_objects.keys())
                
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
                
                
            

                                    

#%% DATA ANALYSIS FROM DATABASE
            elif 'do_data_analysis_from_database' in self.selected_things_to_do:
                
                self.destroy()
                return
                from termcolor import colored
                def print_colored_list(my_list,selected):
                    for i, item in enumerate(my_list):
                        if i in selected:
                            print(colored(item, 'red'))  # Print in red color
                        else:
                            print(item)


                # select mouse
                mousenamesinter=['SPKG','SPHV']
                mousenameschand=['SPKQ','SPKS','SPJZ','SPJF','SPJG','SPKU', 'SPKW','SPKY','SPRB','SPRM','SPRZ','SPSU','SPSM','SPSX','SPSZ']                
                #spKQ chandeliers plane1 18, 27, 121, 228
                #spKQ chandeliers plane2 52(fist pass caiman)
            # INTERNERUIN
                #ALLEN
                #LED OPTOl
                #2p OPTO
                
            # CHandelier
                #LED OPTO
                    # Short dRIFT final
                known_mouse_aq_pair=[[mousenameschand[12],mousenameschand[10]],[[0],[2]]] #ALL LED short drift SPSM, SPRZ
                known_mouse_aq_pair=[[mousenameschand[10]],[[2]]]   #LED short drift SPRZ
                known_mouse_aq_pair=[[mousenameschand[12]],[[0]]]   #LED short drift SPSM
                known_mouse_aq_pair=[[mousenameschand[13]],[[0]]]   #LED short drift SPSX No opto responses
                known_mouse_aq_pair=[[mousenameschand[14]],[[0]]]   #LED short drift SPSZ Control :LED
                known_mouse_aq_pair=[[mousenameschand[13],mousenameschand[14]],[[0],[0]]] #ALL LED short drift  SPSX, SPSZ


                # Short dRIFT BLUE
                known_mouse_aq_pair=[[mousenameschand[9]],[[7]]]   #SPRM   

                    # sPONT blUE
                known_mouse_aq_pair=[[mousenameschand[9]],[[6]]]    #SPRM   

                    
                #2P OPTO
                    # short drift
                known_mouse_aq_pair=[[mousenameschand[9]],[[8,9,10,11]]]    #SPRM   
                known_mouse_aq_pair=[[mousenameschand[10]],[[6,7]]]  #2p short drift SPRZ


                #allen
                known_mouse_aq_pair=[[mousenameschand[2]],[[6,10,17]]]  #this ifor SPJZ three allen
                known_mouse_aq_pair=[[mousenameschand[2]],[[10]]]  #this ifor SPJZ ALlen B

                

#%%

                mousenames,selectedaqposition=known_mouse_aq_pair

                selected_acqs=[]
                for l,mousename in enumerate(mousenames):
                    mouse_object=datamanaging.all_experimetal_mice_objects[mousename]
                    selected_acqs.append(mouse_object.all_mouse_acquisitions)
                    pprint(list(mouse_object.all_mouse_acquisitions.keys()))
                    
                    print_colored_list(list(mouse_object.all_mouse_acquisitions.keys()),selectedaqposition[l])

                assert(len(selected_acqs)==len(selectedaqposition))
                acqs=[]
                for j, mouse in enumerate(selectedaqposition):
                    for i in mouse:
                        acqs.append(selected_acqs[j][list(selected_acqs[j].keys())[i]])
                        
     
               
                # %% get datasets
                selected_analysis=[]
                for i,aq in enumerate(acqs): 
                    # dtset=aq.all_datasets[list(aq.all_datasets.keys())[1]]
                    # dtset.most_updated_caiman.CaimanResults_object.open_caiman_sorter()
                    
                    aq.get_all_database_info() 
                    aq.load_results_analysis(new_full_data=False) 
                    
                    # aq.load_results_analysis(new_full_data=True)       
                    selected_analysis.append({'analysis':aq.analysis_object,'full_data':aq.analysis_object.full_data})

                    print(aq.aquisition_name)
                    
                analysis=selected_analysis[0]['analysis']
                full_data=selected_analysis[0]['full_data']
                
                # self.to_return[5:]=[ mousename,  mouse_object , allacqs, selectedaqposition, acqs, analysis,full_data,selected_analysis]
                
                
                
                #%% combine optodrifting and opto info
                # aq.load_vis_stim_info()
                tt=analysis.signals_object.signal_transitions
                analysis.signals_object.extract_transitions_optodrift('VisStim', led_clipped=True, plot=False)
                
                
                analysis.signals_object.optodrift_info
                
                optograting=analysis.acquisition_object.visstimdict['opto']['randomoptograting']


                #%%
                for i in range(len(selected_analysis)):
                    analysis=selected_analysis[i]['analysis']
                    full_data=selected_analysis[i]['full_data']
                    # analysis.review_aligned_signals_and_transitions()
                    analysis.photostim_stim_table_and_optanalysisrafatemp()
                
                 

                
                # analysis.manually_setting_opto_times_and_cells() # this is for the rafa analysi dataset

              
                
#%% ANALYSIS EXPLORATION

                if 'explore_analysis'  in self.selected_things_to_do:
                    
                      
                    self.destroy()
                    return
                
                    #%% copy to dropbox
                    # datamanaging.copy_data_dir_to_dropbox('SPKG')
    
                    analysis.load_calcium_extractions()
                    res=analysis.caiman_results['220214_SPJZ_FOV1_AllenB_20x_980_52570_narrow_with-000_Plane1_Green']
                    res.dfdt
                    activity_arrays= analysis.get_raster_with_selections('dfdt_raw',plane,selected_cells, paradigm)
    
                    plt.plot( res.dfdt_accepted_matrix[3,:])
                    plt.plot(activity_arrays[0][3,:])
                    #%% variable selections
                    trace_types=['demixed', 'denoised', 
                     'dfdt_raw',  'dfdt_smoothed', 'dfdt_binary',
                     'foopsi_raw', 'foopsi_smoothed','foopsi_binary',
                     'mcmc_raw','mcmc_smoothed','mcmc_binary','mcmc_scored_binary']
                    trace_type=trace_types[-1]
                    final_binary_trace_types=['dfdt_binary','mcmc_binary','mcmc_scored_binary']
    
                    
                    paradigms=['Movie1','Spontaneous','Drifting_Gratings','Movie3','Static_Gratings','Natural_Images','Movie2','Sparse_Noise']
                    paradigm=paradigms[0]
    
                    planes=['All_planes_rough', 'Plane1', 'Plane2', 'Plane3']
                    plane=planes[0]
                    
                    
                    
                    full_raster_pyhton_cell_idx=252
    
                    matlab_sorter_plane='Plane1'
                    matlab_sorter_idx=136
                    
                    full_raster_pyhton_cell_idx_list=[2,12,54,123,201,305]
                    full_raster_pyhton_cell_idx_list=np.arange(0,7)

    
                    selected_cells_options=['All','Pyramidal','Interneurons',full_raster_pyhton_cell_idx_list]
                    selected_cells=selected_cells_options[2]
                    #%% drifitng grating selections
                    
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
                    #%% slicing and indexing cells stimuli planes and cells
                    
                   
                    full_raster_pyhton_cell_idx=1

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
                    full_raster_pyhton_cell_idx_list 
                    if full_raster_pyhton_cell_idx_list.any():
                        all_index_info=[ analysis.convert_full_planes_idx_to_single_plane_final_indx(full_raster_pyhton_cell_idx,plane)  for full_raster_pyhton_cell_idx in full_raster_pyhton_cell_idx_list]
                        
                    # get tomato identity    
                    cell_identity=analysis.indetify_full_rater_idx_cell_identity(full_raster_cell_python_idx, plane)
                    if all_index_info:
                        all_index_info=[(cell, analysis.indetify_full_rater_idx_cell_identity(cell[2], plane))for cell in all_index_info]
                        
                        
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm, drifting_options)
                    #%% general raster by paradigm
                    cells=[2,3,5,6]
                    # activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)

                    trace_type='mcmc_binary'
                    paradigm='Movie1'
                    # analysis.plot_sliced_raster(trace_type,plane,cells,paradigm)
                    analysis.plot_sliced_raster(trace_type,plane,cells,'Full')

                    
                    #%% movie 1 analysis
                    # analysis.extrac_voltage_visstim_signals()
                    
                    analysis.acquisition_object.load_vis_stim_info()
                 
                    movie1table=analysis.movie_one_stim_table()
                    
                    trace_type='mcmc_scored_binary'
                    selected_cells='Interneurons'


                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)
                   
                    # first plot activity for each trial movie
                    
                    
                    plt.close('all')
                    for cell in np.arange(0,7):
                    
                        cell1=np.zeros([10,900])
                        for trial in np.arange(1,11):
                            cell1[trial-1,:activity_arrays[0][cell,np.arange(movie1table[movie1table['Trial_ID']==trial].iloc[0]['start'],movie1table[movie1table['Trial_ID']==trial].iloc[899]['end'])].shape[0]]=activity_arrays[0][cell,np.arange(movie1table[movie1table['Trial_ID']==trial].iloc[0]['start'],movie1table[movie1table['Trial_ID']==trial].iloc[899]['end'])]
                            
                        
                     
                        f,ax=analysis.general_raster_plotting('title')
    
                        ax.imshow( cell1
                                  , cmap='binary', aspect='auto',
                            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
                    
                    trace_type='dfdt_smoothed'
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)
                    plt.close('all')
                    for cell in np.arange(0,7):
                        f,ax=plt.subplots(10,sharex=True)

                        cell1=np.zeros([10,900])
                        for trial in np.arange(1,11):
                            cell1[trial-1,:activity_arrays[0][cell,np.arange(movie1table[movie1table['Trial_ID']==trial].iloc[0]['start'],movie1table[movie1table['Trial_ID']==trial].iloc[899]['end'])].shape[0]]=activity_arrays[0][cell,np.arange(movie1table[movie1table['Trial_ID']==trial].iloc[0]['start'],movie1table[movie1table['Trial_ID']==trial].iloc[899]['end'])]
                            
                            ax[trial-1].plot(cell1[trial-1,:])
                            
                            
                    #%% sta
                    movie1table=analysis.movie_one_stim_table()
                    #plot all activity trcaes for a sinle cell in a given range
                    paradigm= 'Movie1'
                    selected_cells='Interneurons'
                    plane='All_planes_rough'
                    trace_types=[
                        'denoised',
                      # 'dfdt_smoothed',
                      # 'dfdt_binary',                     
                     'mcmc_smoothed',
                     # 'mcmc_scored_binary'
                     ]
                   
                    movistarts= np.arange(0,300,30)
                    plt.close('all')
                    for cell in range(6,7):
                        cellinfo=analysis.convert_full_planes_idx_to_single_plane_final_indx(cell,plane)
                        print(cellinfo)
                        f,ax=plt.subplots(  len(trace_types)+1, figsize=(16,9), dpi=100,sharex=True)
                        f,ax=plt.subplots(  len(trace_types), figsize=(16,9), dpi=100,sharex=True)

                        f.set_tight_layout(True)
                        f.suptitle('Cell: '+str(cell+1))

                        for i,trace_type in enumerate(trace_types):
                            activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)
                            ax[i].plot(activity_arrays[4]-activity_arrays[4][0],activity_arrays[3][cell,:])
                            # ax[i].plot(activity_arrays[3][cell,:])

                            ax[i].set_title(trace_types[i])
                            ax[i].margins(x=0)
                            ax[i].axis('off')

                            # if i==0:
                            #     ax[i].set_ylim([0,40])

                            for trial in np.arange(1,11):
                                
                                fullmoviestartframe=movie1table[movie1table['Trial_ID']==trial]['start'].iloc[0]
                                paradigmsliceframe,=np.where(activity_arrays[5]==fullmoviestartframe)
   
                                ax[i].axvline(movistarts[trial-1],
                                              # ymin=activity_arrays[3][cell,:].min(),
                                              # ymax=activity_arrays[3][cell,:].max(), 
                                              color ='green', lw = 2, alpha = 0.75)
                        
                            # ax[-1].plot(activity_arrays[4][:-1]-activity_arrays[4][0],activity_arrays[-1])
                            # ax[-1].set_title('Running Speed')
                            
                            
                        filename = f'Cell {str(cell+1)} full_movie_recording.pdf'
                        analysis.save_multi_image(filename)
                      
                                    
                    #plot all aligne trials
                    
                    selected_cells='Interneurons'
                    plane='All_planes_rough'
                    trace_types=[
                        'denoised', 
                       # 'dfdt_smoothed', 
                       # 'dfdt_binary',                     
                      'mcmc_smoothed',
                      'mcmc_scored_binary'
                     ]
                    plt.close('all')
                    #%%
                    moviepath=r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\visual_stim\BehaviourCode\AllenStimuli\Smalles\natural_movie_one.mat'

                    movie1_frames=sio.loadmat(moviepath)
                    movie1_frames=movie1_frames['natural_movie_one_all_warped_frames']
                    plt.imshow(movie1_frames[:,:,1])
                    
                    
                    for cell in range(7):
                        cellinfo=analysis.convert_full_planes_idx_to_single_plane_final_indx(cell,plane)
                        print(cellinfo)

                        for trace_type in range(3):
                            f,ax=plt.subplots( 11+1 ,figsize=(16,9), dpi=100,sharex=True)
                            f.suptitle('Cell: '+str(cell+1)+ ' ' +trace_types[trace_type])

                            f.set_tight_layout(True)
                            activity_arrays= analysis.get_raster_with_selections(trace_types[trace_type],plane,selected_cells, paradigm)
                            
                            tracesformean=[]
                            trialtimestamps=[]
                            for trial in range(1,11):
                                trialstartframe,=np.where(activity_arrays[5]== movie1table[movie1table['Trial_ID']==trial]['start'].iloc[0])[0]
                                if trial==10:
                                    trialendframe=len(activity_arrays[5])
                                else:
                                    trialendframe,=np.where(activity_arrays[5]==  movie1table[movie1table['Trial_ID']==trial]['end'].iloc[-1])[0]
                   
                                
                                ax[trial-1].plot(activity_arrays[3][cell,trialstartframe:trialendframe])
                                ax[trial-1].margins(x=0)
                                # if trace_type==0:
                                #     ax[trial-1].set_ylim([0,40])
                                tracesformean.append(activity_arrays[3][cell,trialstartframe:trialendframe])
                                trialtimestamps.append(activity_arrays[4][trialstartframe:trialendframe])
                                
                                    
                            tracesformean[8]=tracesformean[8][:-1]
                            trials=np.vstack(tracesformean)
                                
                            ax[10].set_title('Trial Averaged Activity')

                            ax[10].plot(trials.mean(axis=0),'b')
                            ax[11].set_title('Movie Average Intensity')

                            ax[11].plot(movie1_frames.mean(axis=(0,1)),'r')


                        filename = f'Cell {str(cell+1)} trial_averages.pdf'
                        analysis.save_multi_image(filename)
                        plt.close('all')
                  
                    #get info of a cell indexed from a python raster
                    

                    trial1activity=tracesformean[0]
                    f,ax=plt.subplots( 2, sharex=True)
                    f.set_tight_layout(True)
                    ax[0].plot(trial1activity)
                    ax[1].plot(movie1_frames.mean(axis=(0,1)))
                        
    
                            # DO STA
                    paradigm= 'Movie1'
                    selected_cells='Interneurons'
                    plane='All_planes_rough'
                    trace_type='mcmc_scored_binary'
                    # selected_cells=[1]
                    
                    
                    
                    '''
                    'dfdt_binary',
                    'mcmc_binary',
                    '''
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm)
                    analysis.acquisition_object.load_vis_stim_info() 
                    visstiminfo=analysis.acquisition_object.mat['ops']
                   
                    
                     
                 
                    
                    tosavemovie=cm.movie(movie1_frames)
                    tosavepath=r'C:\Users\sp3660\Desktop\movie1.tiff'
                    tosavemovie.save(tosavepath,order='C')
                    
                    prestimframes=3
                    for cell in range(7):
                        # f,ax=plt.subplots(1,prestimframes)
                        f,ax=plt.subplots(1,figsize=(16,9), dpi=100)
                        

                        # 10 previous frame average
                        spikes, = np.where(activity_arrays[3][cell,:] == 1)
                        spike_full_recording_frames=activity_arrays[5][spikes]
                        
                        stimArray = np.zeros((len(spikes),movie1_frames.shape[0],movie1_frames.shape[1], prestimframes))
                        for j,i in enumerate( spike_full_recording_frames): 
                            if i==65620:
                                i=i+1
                            else:

                                endstim=movie1table[movie1table['start']==i]['Frame_ID'].iloc[0]
                                startstim=endstim-prestimframes
                                stimindexes=np.arange(startstim,endstim)
                                
                                test=movie1_frames[:,:,stimindexes]
                            
                                stimArray[j, :,:,:] =movie1_frames[:,:,stimindexes]
                            
                        print("The stimArray shape is" , stimArray.shape)
                        sta = np.mean(stimArray, axis = (0,-1))
                        # sta = np.mean(stimArray, axis = 0)
                        baselinecorrectedsta=sta-movie1_frames.mean(axis=2)
                        
                        # for image in range(prestimframes):
                        #     ax[image].imshow(sta[:,:,image])
                            
                        ax.imshow(baselinecorrectedsta)
                        filename = f'Cell {str(cell+1)} STA-baseline.pdf'
                        analysis.save_multi_image(filename)
                        plt.show()
                        
                     

    
                    # respondedmov=cm.movie(movie1_frames)
                    # respondedmov=np.moveaxis(respondedmov,[0,2],[2,0])
                    # respondedmov.save('test1.tif',order='F')
                    # respondedmov=cm.movie(movie1_frames)
                    # respondedmov=np.moveaxis(respondedmov,[0,1,2],[2,0,1])
                    # respondedmov.save('test2.tif',order='F')
                    # respondedmov=cm.movie(movie1_frames)
                    # respondedmov=np.moveaxis(respondedmov,[0,2],[2,0])
                    # respondedmov.save('test3.tif',order='C')
                    # respondedmov=cm.movie(movie1_frames)
                    # respondedmov=np.moveaxis(respondedmov,[0,1,2],[2,0,1])
                    # respondedmov.save('test4.tif',order='C')
                    
                
                
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
                    cell=2
                    # trace_type='mcmc_binary'
                    trace_type='mcmc_smoothed'
                    #%%
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
                    
#%% MANUAL VIS STIM INDEXING

                elif 'do_visualstim_indexing'  in self.selected_things_to_do:
                    
                    self.destroy()
                    return
                    analysis.signals_object.process_all_signals()
                    analysis.create_full_data_container()
                    analysis.create_stim_table()
                    
                    #%%movie indexing
                    signals=analysis.signals_object.get_movie_one_trial_structure()
                    
                    
                    
#%% CRF PREP
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




#%% JESUS ENSEMBLES

                elif 'do_jesus_ensemble'  in self.selected_things_to_do:
                    
                    self.destroy()
                    return
                

                    #%% run new jesus analysis
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
                                    
                                    
                    paradigm=paradigms[3]
                    plane = planes[0]
                    for trace_type in final_binary_trace_types[0:]:
                        for selected_cells in selected_cells_options[:3]:
                            activity_arrays= analysis.get_raster_with_selections(trace_type, plane, selected_cells, paradigm=paradigm) 
                            analysis.run_jesus_analysis(activity_arrays)
                            
                    paradigm=paradigms[2]
                    plane = planes[0]
                    trace_type= final_binary_trace_types[-1]
                    for selected_cells in selected_cells_options[:3]:
                        activity_arrays= analysis.get_raster_with_selections(trace_type, plane, selected_cells, paradigm=paradigm) 
                        analysis.run_jesus_analysis(activity_arrays)
                
                    #%% check done jesus results
                    

                    analysis.check_all_jesus_results()
                    pprint(final_binary_trace_types)
                    pprint(planes)
                    pprint(paradigms)
                    pprint(selected_cells_options)
            
                    
                    #%% load jesus results
                    
                    # results get loaded to jesus runs to compare betwen runs
                    analysis.unload_all_runs()
                    analysis.load_jesus_results(sorted_jesus_results['all_cells_grat_mcmcscored'][0])
                    analysis.load_jesus_results(sorted_jesus_results['pyr_grat_mcmcscored'][0])
                    analysis.load_jesus_results(sorted_jesus_results['int_grat_mcmcscored'][0])

            

                    
                    #%% analysis single result run
                    analysis.unload_all_runs()
                    analysis.load_jesus_results(scoredmcmcresults[0])
                    analysis.jesus_runs
                    jesusres_object=analysis.jesus_runs[list(analysis.jesus_runs.keys())[0]]
                    jesusres_object.load_analysis_from_file()
                    jesusanalysis=jesusres_object.analysis
                    jesusoptions=jesusres_object.input_options
  
                    # %UMMARY PLOTTING
                    #%%
                    jesusres_object.plot_raster()
                    jesusres_object.plot_sorted_rasters()
                    jesusres_object.plot_networks()
                    #% THIS ONE IS SLOW WAIT OFR IT
                    jesusres_object.plot_vector_clustering()
                    
                    # ENSEMBLE PLOTTING TO WORK ON
                    act=jesusanalysis['Ensembles']['ActivationSequence']
                    
         
                    
         
                    #%% compare cell typoe runs load runs
                    plt.close('all')
                    plot=0
                    analysis.jesus_runs


                    analysis.unload_all_runs()
                    
                    analysis.load_jesus_results(scoredmcmcresults[0])
                    analysis.load_jesus_results(scoredmcmcresults[1])
                    analysis.load_jesus_results(scoredmcmcresults[2])
                    
                    pyr=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['pyr'][1]).flatten()
                    inter=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['int'][1]).flatten()                   
                    cell_subtype_runs={}
                    for run, run_object in  analysis.jesus_runs.items():
                       
                        ensemble_cell_identity={}
                        jesusanalysis=run_object.analysis
                        if plot:
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
                    
                    
                    #%% plot somthing about ensemble structure and tuning properties
                        #plot ensmebles cell rois
                        # find where each ensembles is activ
                        # find which stimulu is associated with a given ensmeble
                        # try to decode stimulus based on sensemble
                    import itertools

                    aallplanescellids=list(itertools.chain.from_iterable([ list(map( lambda x:(list((x,plane)))  ,full_data['imaging_data']['All_planes_rough']['CellIds'][plane] )) 
                                                                          for plane in full_data['imaging_data']['All_planes_rough']['CellIds']]))
                    
                    
                    
                    
                    # planes=['Plane1', 'Plane2', 'Plane3']
                    # full_data['imaging_data'][plane]['CellIds']
                    
                    
                    
                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]
                    #%%
                    from scipy.ndimage import gaussian_filter
                    plt.close('all')
                    
                    methods=['combined', 'pyramidals', 'interneurons']

                    
                    allensembles=[combined, pyramidals, interneurons]
                    cell_types=['Pyramidals','Interneurons']
                    planes=['Plane1','Plane2','Plane3']
                    for m, meth in enumerate(allensembles):
                        for key, ensemble in meth[1].items():
                            allcells=[[ sorted([aallplanescellids[i][0] for i in ensemble[cell_type]  if aallplanescellids[i][1]==plane]) for plane in planes] for cell_type in cell_types]
                            allceltypesmask=[]
                            for cell_type in allcells:
                                
                                allplanesmasks=[]
                                for plane in cell_type:
                
                                    allmask=np.zeros(dims)
                                    for i in plane:
                                        allmask=allmask+ np.reshape(res.data['est']['A'][:,i-1].toarray(), dims, order='F')
                                        
                                    allmask[allmask<70]=0
                                    allmask[allmask>70]=255
                                    allmask=gaussian_filter(allmask,0.5)
                                    allmask[allmask>0]=255
                                    allplanesmasks.append(allmask)
                                    
                                allceltypesmask.append(allplanesmasks)
           
                                                
                            # fig    = plt.figure()
                            # ax     = fig.gca(projection='3d')
                            
                            x      = np.arange(allplanesmasks[0].shape[0])
                            X, Y   = np.meshgrid(x, x)
                            # levels=[254,255]
                            # for i, mask in enumerate(allceltypesmask[0]):
                                
                            #     ax.contourf(X, Y, mask, levels,
                            #             colors=('k'),
                            #             zdir='z', offset=1.5*i, alpha=1)
                                
                            #     ax.contourf(X, Y, allceltypesmask[1][i], levels,
                            #             colors=('r'),
                            #             zdir='z', offset=1.5*i, alpha=1)
                               
                            
                            # ax.set_zlim3d(0, 3.5)
                            # plt.grid(False)                                       
                            # ax.view_init(elev=25., azim=30)

                            f,ax=plt.subplots(1,3, figsize=(20,9))
           
                            levels=[254,255]
                            for i, mask in enumerate(allceltypesmask[0]):
                                ax[i].contourf(X, Y, mask, levels,
                                           colors=('k'),
                                            )
                                
                                ax[i].contourf(X, Y, allceltypesmask[1][i], levels,
                                        colors=('r'),
                                          )
                                ax[i].axis('square')
                            plt.tight_layout()
                            f.suptitle(f'{key.replace(": ", "_")} Composition')

                            # plt.show()
                            
                            
                                
                            filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_{methods[m]}_{key.replace(": ", "_")}_ensemble_components.pdf')
                            run_object.save_multi_image(filename)

                                                            
                    #%% compare different cell type ensemble similarities
                    
                    plt.close('all')

                    combinedraster=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]][2].astype('float')
                    pyramidalraster= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]][2].astype('float')
                    interneuronraster= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]][2].astype('float')
                    
                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]
                        
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
                    
                    
                    fig,ax=plt.subplots(1,1, figsize=(20,9))
                    pyr_ad=ax.imshow(adjacency_pyr, cmap='jet', aspect='auto')
                    fig.colorbar(pyr_ad, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
                    fig.suptitle('Ensemble Cosine Similarity')
                    ax.set_ylabel('Combined Ensembles')
                    ax.set_xlabel('Pyramidal Ensembles')
                    plt.tight_layout()



                    filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_pyr_ensemble_similarity.pdf')
                    run_object.save_multi_image(filename)

                    plt.close('all')
                    
                    
                    
                    
                    # most similar ensembles
                    pyramid_most_similars=[]    
                    for i,combined_ensemble in enumerate(adjacency_pyr):
                        pyramid_most_similars.append([i+1, combined_ensemble.argmax()+1,combined_ensemble.max()])
                        
                    pyr_equivalents=[i for i in pyramid_most_similars if i[2]>0.1]   
                           
                    inter_most_similars=[]    
                    for combined_ensemble in adjacency_inter:
                        inter_most_similars.append([combined_ensemble.argmax()+1,combined_ensemble.max()])
                                

                 

                    
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
                           
                     # MOST SIMILAR ENSEMBLES
                  
                           



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
                        
                    filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_ensemble_by_ensemble.pdf')
                    combined[0].save_multi_image(filename)

                    plt.close('all')
                    
                    
                     
                        



                    #%% ensemble comparison betwen three cell type runs

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
                    ensmeble='Ensemble: 3'
                    
                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]
                    
                    
                    celtyp='Pyramidals'
                    ense=pyramidals[1][ensmeble][celtyp]
                    
                    # celtyp='Interneurons'
                    # ense=interneurons[1][ensmeble][celtyp]
                    
                    cells=np.array(ense)
                    
                    
                    
                    # celtyp='All'
                    # cells=np.concatenate([np.array(ensemble_cell_identity[ensmeble]['Pyramidals']).astype('int'), np.array(ensemble_cell_identity[ensmeble]['Interneurons']).astype('int')])

                   

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
                    
                    #%%       
                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]

                    
                    selectedtun=interneurons
                    
                    plt.close('all')
                    for idx in range(len(selectedtun[1].keys())):
                        print(selectedtun[1][list(selectedtun[1].keys())[idx]])

                    idx=1
                    ensmeble=list(selectedtun[1].keys())[idx]
                    ens=selectedtun[0].analysis['Ensembles']

                    allcells=selectedtun[1][ensmeble]
                    ens['VectorID']
                    ens['EnsembleNeurons'][idx]
                    seq=ens['ActivationSequence']
                    activation=ens['ActivityBinary']
                    
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm, drifting_options)

                    traces=activity_arrays[2][ens['EnsembleNeurons'][idx],:]
                    cells=ens['EnsembleNeurons'][idx]
                    meanacti=[]
                    cell=389
                    for cell in cells:
                        s2,_=analysis.plot_orientation(cell,trace_type,plane)
                        meanacti.append((s2))
                        
                        analysis.plot_blank_sweeps(cell, trace_type, plane)
                    #%%

                    analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']
                    
                    indexes=((analysis.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_first'],
                    analysis.full_data['visstim_info']['Paradigm_Indexes']['first_drifting_set_last']),
                    (analysis.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_first'],
                    analysis.full_data['visstim_info']['Paradigm_Indexes']['second_drifting_set_last']),
                    (analysis.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_first'],
                    analysis.full_data['visstim_info']['Paradigm_Indexes']['third_drifting_set_last'])) 
                    
                    # tranfrorm situmlus table to 0nly grtaing frames
                    
                    firstgrat=analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']['start']<indexes[1][0]
                    secondgrat=(analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']['start']>indexes[0][1]) & (analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']['start']<indexes[2][0])
                    thirdgrat=analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table']['start']>indexes[1][1]
                    
                    
                    firssubtract=indexes[0][0]
                    secondsubstract=indexes[1][0]-indexes[0][1]
                    thirdsubstract=indexes[2][0]-indexes[1][1]
                    
                    netable=   analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'].copy()
        
                    netable['start']=netable['start']-firssubtract
                    netable['end']=netable['end']-firssubtract
                    
                    
                    netable['start'][netable['start']>indexes[0][1]]=netable['start'][netable['start']>indexes[0][1]]-secondsubstract
                    netable['end'][netable['end']>indexes[0][1]]=netable['end'][netable['end']>indexes[0][1]]-secondsubstract

                    netable['start'][netable['start']>(indexes[1][1]-secondsubstract)]=netable['start'][netable['start']>(indexes[1][1]-secondsubstract)]-thirdsubstract
                    netable['end'][netable['end']>(indexes[1][1]-secondsubstract)]=netable['end'][netable['end']>(indexes[1][1]-secondsubstract)]-thirdsubstract

                    substim=netable[(netable['orientation']==90)]
                    
                    
                    pixel_per_bar = 4
                    dpi = 100
                
                    fig, ax = plt.subplots(1,  figsize=(16,9), dpi=dpi, sharex=True)
                    ax.imshow(activation, cmap='binary', aspect='auto',
                        interpolation='nearest', norm=mpl.colors.Normalize(0, 0.01))
                    for j in substim['start'].values:
            
                        ax.axvspan(xmin=j , xmax=j+33, color='red', alpha=0.2)


                    

                    
                    #%%
                    
                    ensemble=list(similarities.keys())[1]   
                    cells=copy.copy(combined[1][ensemble]['Pyramidals'])
                    cells.extend(combined[1][ensemble]['Interneurons'])
                    analysis.plot_orientation(391,trace_type,plane,plot=True)

                    # for cell in cells:
                    #     #%%
                    #     correctedcell=analysis.do_some_plotting(cell, trace_type, plane)
                    #     #%%
                    #     analysis.plot_orientation(correctedcell,trace_type,plane)
                    # #%%
                    
                    
                    all_mena_act=[]
                    cells=copy.copy(pyramidals[1][ensmeble]['Pyramidals'])
                    meanacti=[]
                    for cell in cells:
                        s2,_=analysis.plot_orientation(cell,trace_type,plane)
                        meanacti.append((s2))


                    meanoriactensemble=np.zeros((4,1))
                    test=list(zip(*meanacti))
                    meani=[]
                    for j,i in enumerate(test):
                        meanoriactensemble[j]=(np.mean(i))
                    all_mena_act.append(meanoriactensemble)
                    
                    
                    
                    for ensemble in similarities['Ensemble: 1']['Pyramidals'].keys():
                        cells=copy.copy(pyramidals[1][ensemble]['Pyramidals'])
                        meanacti=[]
                        for cell in cells:
                            s2,_=analysis.plot_orientation(cell,trace_type,plane)
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
                            s2,_=analysis.plot_orientation(correctedcell,trace_type,plane)
                            meanacti.append((s2))
    
    
                        meanoriactensemble=np.zeros((4,1))
                        test=list(zip(*meanacti))
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
                        
                  

#%% TUNING ALLEN ANALYSIS
                if 'do_tuning_allen'  in self.selected_things_to_do:
                    analysis.load_allen_analysis()
                    self.destroy()
                    return
                    #%% creqating sweep response based on selections
                    print( trace_types)
                    print(planes)
                    print(paradigms)
                    print(selected_cells_options)
                    plane=planes[0]
                    matrix=trace_types[-1]
                    allen_mock=analysis.allen_analysis
                    allen_mock.set_up_drifting_gratings_parameters()
                    allen_mock.do_AllenA_analysis(plane, matrix)
                    allen_mock.response=allen_mock.get_response()
                    response=allen_mock.response
                    
                    a=allen_mock.mean_sweep_response
                    aa=allen_mock.sweep_response
                    aaa=response
                    
                 
                   
                    
                    
                    #%%
                    allen_mock.peak=allen_mock.get_peak()
                    peak=allen_mock.peak
                    
                    
                    activity_arrays= analysis.get_raster_with_selections(trace_type,plane,selected_cells, paradigm, drifting_options)
                    tom=pd.DataFrame([cell[1] for cell in activity_arrays[-2][3]])
                    
                    peak['Tomato'] = tom
                 
                    
                    
                    peakinter=peak[peak['Tomato']=='Tomato +']
                    peakpyr=peak[peak['Tomato']=='Tomato -']
                    peak_dff_min=0.02
                    peakfilteredinactive=peak[peak.peak_dff_dg>peak_dff_min]
                    
                    peakinterfiltered=peakfilteredinactive[peakfilteredinactive['Tomato']=='Tomato +']
                    peakpyrfiltered=peakfilteredinactive[peakfilteredinactive['Tomato']=='Tomato -']


                    #%% single cell orientation tuning
                    cell=391
                    included=combined[1]['Ensemble: 5']['Pyramidals']
                    excludedcells=set(pyramidals[1]['Ensemble: 5']['Pyramidals'])^ set(combined[1]['Ensemble: 4']['Pyramidals'])
                    cells=included
                    
                    for cell in cells:
                        allen_mock.open_star_plot(include_labels=True,cell_index=cell,show=True)
                        s2,_=analysis.plot_orientation(cell,trace_type,plane,plot=True)
                        # analysis.plot_blank_sweeps(cell, trace_type, plane)
                    #%% find unique and promiscous enxsemble cells
                    import itertools

                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]

                    
                    selectedtun=interneurons
                    sets=[]
                    for idx in range(len(selectedtun[1].keys())):
                       sets.append(set(selectedtun[1][list(selectedtun[1].keys())[idx]]['Interneurons']))
                    fullsharedcells=[]
                    for i,seti in enumerate(sets):
                        print(f'Ensemble: {i} cell number {len(seti)}')
                        print(seti)
                        totalsharedcels=[]

                        for j, s in enumerate(sets):
                            if j!=i:
                                inter=seti.intersection(s)
                                totalsharedcels.append(inter)
    
                                print(f'Ensemble: {j} shared {len(inter)}')
                                
                                


                        flatten_list = list(itertools.chain(*totalsharedcels))

                        fullsharedcells.append((set(flatten_list), len(flatten_list)/len(seti))   )        
                        
                        
                    
                    #%% all ensemble cell orientation with jesus ensembles
 
                    
                    combined=cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_All_jesus' in i][0]]
                    pyramidals= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Pyramidal_' in i][0]]
                    interneurons= cell_subtype_runs[[ i for i in cell_subtype_runs.keys() if '_Interneurons_' in i][0]]

                    
                    runs_names=['Combined','Pyramidal','Interneurons']
                    runs=[combined,pyramidals,interneurons]
                    celltypes=['Pyramidals', 'Interneurons']
                    orientation=1
                    direction=1
                    plot_cells=0
                    directions=np.linspace(0,360-45,8).astype('uint16')
                    angles=directions[:4]
                    
                    
                    meth_orientations=[]
                    for i, run in enumerate(runs):
                        
                        plt.close('all')
                        for ensemble in range(len(run[1].keys())):
                            print(run[1][list(run[1].keys())[ensemble]])
                            
                            
                        celltypes_orientations=[]
                        for cell_type in celltypes:
       
                            ensemble_orientations=[]                         
                            for ensemb in list(run[1].keys()):
                                if run[1][ensemb][cell_type]:

                                    cells= sorted(run[1][ensemb][cell_type])
                                    if plot_cells:
                                        for cell in cells:
                                            allen_mock.open_star_plot(include_labels=True,cell_index=cell, show=None)
                                        
                                    ensemble_orientation=peak.iloc[cells]['ori_dg'].values
                                    ensemble_orientations.append(ensemble_orientation)
                        
                            celltypes_orientations.append(ensemble_orientations)
                                        
                        meth_orientations.append(celltypes_orientations)
                        
                        
                    meth_orientations[0].append([np.hstack([ensmeble, meth_orientations[0][1][i]]) for i , ensmeble in enumerate(meth_orientations[0][0])])
                    
                    for i, run in enumerate(meth_orientations):
                
                        for j, cell_type in enumerate(run):
                            if j<2:
                                newcelltypes=celltypes[j]
                                
                            else:
                                newcelltypes='Pyr + Int'

                            for k, ensemb in enumerate(cell_type):
                               
                                    # cells= ensemb
                                    # if plot_cells:
                                    #     for cell in cells:
                                    #         allen_mock.open_star_plot(include_labels=True,cell_index=cell, show=None)
                                        
                                if orientation:
            
                                    f,ax=plt.subplots(1, figsize=(20,9))
                
                                    mu, std = norm.fit ([i-4 if i>3 else i for i in ensemb])
                                    ax.hist([i-4 if i>3 else i for i in ensemb ], bins=range(5),density=True, alpha=0.6, align='left')
                                    ax.set_xticks(ticks=range(4))
                                    ax.set_xticklabels(labels=angles)

                                    xmin, xmax = ax.get_xlim()
                                    x = np.linspace(xmin, xmax, 100)
                                    p = norm.pdf(x, mu, std)
                                    ax.plot(x, p, 'k', linewidth=2)
                                    ax.set_ylim(0,1.2)
                                    f.suptitle(runs_names[i] +' '+ list(runs[i][1].keys())[j] +' ' + newcelltypes)
              
                                    
                                    
                                if direction:
                                    f,ax=plt.subplots(1, figsize=(20,9))
   
                                    mu, std = norm.fit(ensemb.tolist())
                                    ax.hist( ensemb , bins=range(9),density=True, alpha=0.6, align='left')
                                    ax.set_xticks(ticks=range(8))
                                    ax.set_xticklabels(labels=directions)
                                    xmin, xmax = ax.get_xlim()
                                    x = np.linspace(xmin, xmax, 100)
                                    p = norm.pdf(x, mu, std)
                                    ax.plot(x, p, 'k', linewidth=2)
                                    ax.set_ylim(0,1.2)
                                    f.suptitle(runs_names[i] +' '+ list(runs[i][1].keys())[j] +' ' + newcelltypes)

                                    # plt.hist(peak.iloc[cells]['reliability_dg'])
                                    # plt.show()
                                    # plt.hist(peak.iloc[cells]['osi_dg'])
                                    # plt.show()
            
                                    # plt.hist(peak.iloc[cells]['dsi_dg'])
                                    # plt.show()
            
                                    # plt.hist(peak.iloc[cells]['peak_dff_dg'])
                                    # plt.show()
            
                                    # plt.hist(peak.iloc[cells]['ptest_dg'])
                                    # plt.show()
            
                                    # plt.hist(peak.iloc[cells]['cv_os_dg'])
                                    # plt.show()
            
                                    # plt.hist(peak.iloc[cells]['cv_ds_dg'])
                            
                                filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_{runs_names[i]}_{list(runs[i][1].keys())[k].replace(": ", "_")}_{newcelltypes}_ensemble_grating_selectivities.pdf')
                                run_object.save_multi_image(filename)
                                                        
                    #%% other parameter exploration global  
                    plt.hist(peak['ptest_dg'].values)
                    plt.show()

                    plt.hist(100*peak['peak_dff_dg'].values)
                    plt.show()
                    
                    plt.hist(100*peak['reliability_dg'].values) # PREFERED CONDIION RELIABILITY
                    plt.show()

                    nonsignifantcells=peak[peak['ptest_dg'].values>0.05]
                    significantcells=peak[peak['ptest_dg'].values<0.05]
                    percentage_stim_responsive_cells=100*significantcells.shape[0]/peak.shape[0]
                    
                    
                    runs=[combined,pyramidals,interneurons]
                    runs_names=['Combined','Pyramidal','Interneurons']
                    celltypes=['Pyramidals', 'Interneurons']

                    
                    
                    #%% other parameter loop all ensembles
                    
                    runs=[combined,pyramidals,interneurons]
                    runs_names=['Combined','Pyramidal','Interneurons']
                    celltypes=['Pyramidals', 'Interneurons']

                    run_idx=0
                    run=runs[run_idx][1]
                    ensesmblemnumbers=list(range(10))
                    ens_idx=1
                    ensemble=f'Ensemble: {ensesmblemnumbers[ens_idx]}'
                                      
                    print( trace_types)
                    print(planes)
                    print(paradigms)
                    print(selected_cells_options)
                    trace_type=trace_types[-1]
                    
                    all_freq_means=[]
                    for i, r in enumerate(runs):
                        frequencies_means=[]

                        ensemble_struc=r[1]
                        for ensemble_idx in range(1,len(r[1])+1):
                            ensemble=f'Ensemble: {ensesmblemnumbers[ensemble_idx]}'
                            for cell_type in celltypes:
                                if ensemble_struc[ensemble][cell_type]:
                                    cells=ensemble_struc[ensemble][cell_type]
                                    print('_'.join([runs_names[i],ensemble,cell_type]))
                                    print(peak.iloc[cells]['ptest_dg'].values<0.05)
                                    print(100*sum(peak.iloc[cells]['ptest_dg'].values<0.05)/len(peak.iloc[cells]['ptest_dg'].values<0.05))
                                    
                                    # plt.hist(peak.iloc[cells]['ptest_dg'].values)
                                    # plt.show()
                                    # plt.hist(100*peak.iloc[cells]['peak_dff_dg'].values)
                                    # plt.show()

                # calculate mean cell reliability per ensemble
                                    labels=['reliability_dg','osi_dg','cv_os_dg','dsi_dg','cv_ds_dg','tf_index_dg']
                                    params_to_print=[np.mean(peak.iloc[cells]['reliability_dg']),
                                    
                                                        np.mean(peak.iloc[cells]['osi_dg']),
                                                        np.mean(peak.iloc[cells]['cv_os_dg']),
                    
                                                        np.mean(peak.iloc[cells]['dsi_dg']),
                                                        np.mean(peak.iloc[cells]['cv_ds_dg']),
                                                        
                                                        np.mean(peak.iloc[cells]['tf_index_dg'])]
                                    
                                    frequencies_means.append( np.mean(peak.iloc[cells]['dsi_dg']))
                                    
                                    
                                    for k,j in enumerate(params_to_print):
                                        print(labels[k])
                                        print(j)
                
                        all_freq_means.append(frequencies_means)
                        
                    #%%
                    x_sharp= np.arange(0, 2*np.pi+2*np.pi/8, 2*np.pi/8)
                    X_ = np.linspace(np.radians(directions).min(),2*np.pi, 500)
                    
                    for i, r in enumerate(runs):
                       frequencies_means=[]

                       ensemble_struc=r[1]
                       for ensemble_idx in range(1,len(r[1])+1):
                           ensemble=f'Ensemble: {ensesmblemnumbers[ensemble_idx]}'
                           combinedcells=[]
                           for cell_type in celltypes:
                               if ensemble_struc[ensemble][cell_type]:
                                   cells=ensemble_struc[ensemble][cell_type]
                                   combinedcells=combinedcells+cells
                                   
                                   mm=all_angle_mean_reponses[np.array(sorted(cells)),:].mean(axis=0)
                                   mm_std=all_angle_mean_reponses[np.array(sorted(cells)),:].std(axis=0)
                                   joined_mm=np.append(mm,mm[0])
                                   joined_mm_std=np.append(mm_std,mm_std[0])

  
                                   f,ax=plt.subplots(1, figsize=(20,9), subplot_kw={'projection': 'polar'})
                                   ax.plot(x_sharp, joined_mm)

                                   ax.errorbar(x_sharp, joined_mm, joined_mm_std, linestyle='None', marker='^')

                                 
                                   # X_Y_Spline = make_interp_spline(x_sharp, joined_mm)                        
                                   # Y_ = X_Y_Spline(X_)
                                   # ax.plot(X_, Y_)
                                  
                                   f.suptitle('_'.join([runs_names[i],ensemble,cell_type]))
                                   ax.set_xticklabels(directions) 
                                   ax.set_ylim(-0.05,0.25)
                                   
                                   filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_{runs_names[i]}_{ensemble.replace(": ", "_")}_{cell_type}_polar_ensemble_grating_Dff.pdf')
                                   run_object.save_multi_image(filename)
                                   
                           if i==0 and cell_type=='Interneurons':
                                     
                                 mm=all_angle_mean_reponses[np.array(sorted(combinedcells)),:].mean(axis=0)
                                 mm_std=all_angle_mean_reponses[np.array(sorted(cells)),:].std(axis=0)
                                 joined_mm=np.append(mm,mm[0])
                                 joined_mm_std=np.append(mm_std,mm_std[0])
      
      
                                 
                                 f,ax=plt.subplots(1, figsize=(20,9), subplot_kw={'projection': 'polar'})
                                 ax.plot(x_sharp, joined_mm)
                                 ax.errorbar(x_sharp, joined_mm, joined_mm_std, linestyle='None', marker='^')
      
      
                                 
                                 # X_Y_Spline = make_interp_spline(x_sharp, joined_mm)                        
                                 # Y_ = X_Y_Spline(X_)
                                 # ax.plot(X_, Y_)
      
                                 
                                 
                                 f.suptitle('_'.join([runs_names[i],ensemble,'full_cells']))
                                 ax.set_xticklabels(directions) 
                                 ax.set_ylim(-0.05,0.25)
                            
                   
                                 filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_{runs_names[i]}_{ensemble.replace(": ", "_")}_Pyr + Int_polar_ensemble_grating_Dff.pdf')
                                 run_object.save_multi_image(filename)
                                   


                    #%% comparing exc equivalent ensembles   
                    all_comsp=[]
                    parameters=[col for col in peak.columns]

                    for ensmeble_simils in pyr_equivalents:
                        ensmeb=runs[0][1][f'Ensemble: {ensmeble_simils[0]}']
                        onlypyramidalcell=ensmeb['Pyramidals']
                        full=pyramidalcell+ensmeb['Interneurons']
                        pyrensemb=runs[1][1][f'Ensemble: {ensmeble_simils[1]}']
                        pyramidalcells=pyrensemb['Pyramidals']
                        
                        
                        mixed={'pyr_only':pyramidalcells, 'combined_pyr':onlypyramidalcell, 'combined_full':full }
                        
                        
                        equivalent_comparisons={k:{parameter: np.mean(peak.iloc[v][parameter]) for parameter in parameters[:-1]} for k,v in mixed.items()}
                        all_comsp.append(equivalent_comparisons)   
                        
                    plt.bar(equivalent_comparisons.keys(),[v['reliability_dg'] for k,v in equivalent_comparisons.items()])
                    
                    width = 0.2 
                    labels=[f'Combined Ensmeble: {i[0]}' for i in pyr_equivalents]
                    x = np.arange(1,len(labels)+1)
                    labels.insert(0,0)

                    for parameter in parameters[:-1]:
                        
                        Pyr_Only=[i['pyr_only'][parameter] for i in all_comsp]
                        Combined_Pyr=[i['combined_pyr'][parameter] for i in all_comsp]
                        Combined_Full=[i['combined_full'][parameter] for i in all_comsp]
                        
                        
                        fig, ax =plt.subplots(1, figsize=(20,9))
                        ax.bar(x - 0.2, Pyr_Only,     width,      label='Pyr_Only')
                        ax.bar(x, Combined_Pyr,  width,  label='Combined_Pyr')
                        ax.bar(x + 0.2, Combined_Full, width, label='Combined_Full')
                        ax.set_xticklabels(labels)
                        ax.legend(["Pyr_Only", "Combined_Pyr", "Combined_Full"])
                        fig.suptitle(parameter)
                        
                    
                    filename = os.path.join(os.path.split(combined[0].analysis_path)[0],f'{"_".join(combined[0].input_options)}_{combined[0].timestr}_ensemble_grating_parameter_comp.pdf')
                    run_object.save_multi_image(filename)
                                            
                            
                        
                  
                        
                        
                     
                    #%% single cell
                    run_idx=0
                    run=runs[run_idx][1]
                    ensesmblemnumbers=list(range(10))
                    ens_idx=1
                    ensemble=f'Ensemble: {ensesmblemnumbers[ens_idx]}'
                    
                    pyr_cells=run[ensemble][celltypes[0]]
                    int_cells=run[ensemble][celltypes[1]]
                    
                    cell=pyr_cells[5]
                    if peak.iloc[cell]['ptest_dg']<0.05:
                        s2,_=analysis.plot_orientation(cell,trace_type,plane,plot=True)
                        allen_mock.open_star_plot(include_labels=True,cell_index=cell, show=True)
                        pprint(peak.iloc[cell])
                    else:
                        print('not significantly tuned')
                        


                     
                    all_angle_mean_reponses=np.zeros([response.shape[2], response.shape[0]])
                    for i in range(response.shape[0]):
                    
                        all_angle_mean_reponses[:,i]=response[i, 1:, :, 0].mean(axis=0)
                        
          
                    
                    
                 

                    
                      

                    
                    
                    #%% general population properties of tuning single cells
                    celltypes=['All', 'Tomato +', 'Tomato -']
                    celltype=celltypes[2]
                    for celltype in celltypes:
                        allen_mock.plot_orientation_selectivity(peak_dff_min=0.02, cell_type=celltype)
                    for celltype in celltypes:
                        allen_mock.plot_direction_selectivity(peak_dff_min=0.02,cell_type=celltype)
                    for celltype in celltypes:
                        allen_mock.plot_preferred_direction(peak_dff_min=0.02, cell_type=celltype)
                    for celltype in celltypes:
                        allen_mock.plot_preferred_temporal_frequency(peak_dff_min=0.02, cell_type=celltype)

                    allen_mock.signal_noise_correlations()
                    

                
                    #%% pca
             
                
                    print( trace_types)
                    print(planes)
                    print(paradigms)
                    print(selected_cells_options)
                    plane=planes[0]
                    matrix=trace_types[-1]
                    cell_selecton_index=2
                    cell_type=selected_cells_options[cell_selecton_index]

                    pyr=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['pyr'][1]).flatten()
                    inter=np.argwhere(analysis.pyr_int_ids_and_indexes['All_planes_rough']['int'][1]).flatten()     
                    all_cells=np.concatenate((pyr, inter))
                    cells=(all_cells,pyr,inter)
                    
                    
                    analysis.allen_analysis.do_AllenA_analysis(plane, matrix)
                    selectedcells=cells[cell_selecton_index]
                    params=(matrix,plane,cell_type)
                    #%%
                    analysis.do_PCA(analysis.allen_analysis.mean_sweep_response.iloc[:,selectedcells], analysis.allen_analysis.sweep_response.iloc[:,selectedcells], analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'], params)

                    #%%
                    
                

    
                    
                    # for matrix in  trace_types[8:]+ trace_types[3:5]:
                    #     for i, cell_type in enumerate( selected_cells_options[:-1]):
                    #         analysis.allen_analysis.do_AllenA_analysis(plane, matrix)
                    #         selectedcells=cells[i]
                    #         params=(matrix,plane,cell_type)
                    #         analysis.do_PCA(analysis.allen_analysis.mean_sweep_response.iloc[:,selectedcells], analysis.allen_analysis.sweep_response.iloc[:,selectedcells], analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'], params)

                       
                    matrix=trace_types[-1]

                    for i, cell_type in enumerate( selected_cells_options[:-1]):
                        analysis.allen_analysis.do_AllenA_analysis(plane, matrix)
                        selectedcells=cells[i]
                        params=(matrix,plane,cell_type)
                        analysis.do_PCA(analysis.allen_analysis.mean_sweep_response.iloc[:,selectedcells], analysis.allen_analysis.sweep_response.iloc[:,selectedcells], analysis.full_data['visstim_info']['Drifting_Gratings']['stimulus_table'], params)
                         
#%% YURIY ENSEMBLE ANALYSIS

                if 'do_yuriy_ensembles'  in self.selected_things_to_do:
                    analysis.load_yuriy_analysis()
                    self.destroy()
                    return
   
#%% TRANFER DATA TO ANALYSIE OFFSITE                
        elif [todo for todo in self.selected_things_to_do if 'cloud' in todo]:
            
            lab=self.projectManager.initialize_a_project('LabNY', self.gui)   
            MouseDat=lab.database
            
            self.to_return=[lab, [],MouseDat,datamanaging,[],[], [],[],[],[], [],[],[]]
            
            # select mouse
            mousenamesinter=['SPKG','SPHV']
            mousenameschand=['SPKQ','SPKS','SPJZ','SPJF','SPJG','SPKU', 'SPKW','SPKY']
            
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
            
            
            from ny_lab.data_analysis.resultsAnalysis import ResultsAnalysis
            allen_results_analysis=ResultsAnalysis(allen_BO_tuple=(allen,data_set,spikes ))
            
            
            analysis=acq.analysis_object
            full_data=analysis.full_data
        
            print(acq.aquisition_name)
            self.to_return[5:]=[ mousename,  mouse_object , allacqs, selectedaqposition, acq, analysis,full_data]
            
            
            
            
            
            pass
    if __name__ == "__main__":
        
        
        pass    