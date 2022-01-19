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




class ProjectManager(Project):    
    def __init__(self, githubtoken_path, computer, platform):
        self.githubtoken_path=githubtoken_path
        Project.g = Github(open(githubtoken_path, 'r').read())
        Project.u = Project.g.get_user()
        Project.all_github_repos={}
    
        # intitialize some variables for all projects
        Project.all_paths_for_this_system={}
        Project.all_dir=['Github','Documents','Dropbox']

        Project.__init__(self, 'ProjectManager', githubtoken_path, computer, platform)
      
        self.main_directory=self.project_paths['Documents']  
        
        
        if Project.platform=='win32':
            self.all_projects_in_disk={os.path.split(direc)[1]:direc for direc in glob.glob(self.all_paths_for_this_system['Github']+'\\**')  if os.path.isdir(direc)}

        elif Project.platform=='linux':
        
            self.all_projects_in_disk={os.path.split(direc)[1]:direc for direc in glob.glob(self.all_paths_for_this_system['Github']+'/**')  if os.path.isdir(direc)}

    def clone_project(self, project):

        empty=Project.solve_github_repo(self, project=project)

    
    def initialize_a_project(self, project, gui) :
        if project=='LabNY':
            sys.path.insert(0,  self.all_projects_in_disk['LabNY'])
            # # sys.path.insert(0, os.path.join(self.all_projects_in_disk['LabNY'],' ny_lab'))
            sys.path.insert(0, self.all_projects_in_disk['ProjectManager'])
            import ny_lab
            self.lab=ny_lab.RunNYLab( self.githubtoken_path, gui)
            return self.lab
        elif project=='AllenBrainObservatory':
            sys.path.insert(0,  self.all_projects_in_disk['AllenBrainObservatory'])
            # # sys.path.insert(0, os.path.join(self.all_projects_in_disk['LabNY'],' ny_lab'))
            sys.path.insert(0, self.all_projects_in_disk['ProjectManager'])
            import allen
            self.allen_ob=allen.AllenBrainObservatory(self.githubtoken_path)
            return self.allen_ob
                   








    
        
