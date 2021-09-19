# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:28:30 2021

@author: sp3660
"""



from github import Github
from .ProjectsCLass import Project
import os
import glob
import sys
sys.path.insert(0, r'C:/Users/Samuel/Documents/Github/LabNY')
sys.path.insert(0, r'C:/Users/Samuel/Documents/Github/LabNY/ny_lab')
import ny_lab



class ProjectManager(Project):    
    def __init__(self, githubtoken_path, computer, platform):
        
        Project.g = Github(open(githubtoken_path, 'r').read())
        Project.u = Project.g.get_user()
        Project.all_github_repos={}
    
        # intitialize some variables for all projects
        Project.all_paths_for_this_system={}
        Project.all_dir=['Github','Documents','Dropbox']

        Project.__init__(self, 'ProjectManager', githubtoken_path)
      
        self.main_directory=self.project_paths['Documents']  
        
        self.all_projects_in_disk=[os.path.split(direc)[1] for direc in glob.glob(self.all_paths_for_this_system['Github']+'\\**')  if os.path.isdir(direc)]
        
        
        
    def initialize_a_project(self, project, gui) :
        if project=='LabNY':
            self.lab=ny_lab.RunNYLab(gui)
            return self.lab
            








    
        
