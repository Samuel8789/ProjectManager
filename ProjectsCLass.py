# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
This is code to stablish the project class and control all paths in any computer and get those 
paths for any given project.
The basic structure is
    Read Github repos


"""
import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/ProjectManager')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/ProcessingScripts')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses/Mouse Managing')

from select_values_gui import select_values_gui
import ctypes.wintypes
import os, string
from pathlib import Path
import json
from platform import system
import git
from git import Repo
from github import Github

class Project:
    # get github info
    g = Github(open(r'C:\Users\sp3660\Documents\Github\GitHubToken.txt', 'r').read())
    u = g.get_user()
    all_github_repos={}

    # intitialize some variables for all projects
    all_paths_for_this_system={}
    all_dir=['Github','Documents','Dropbox']
      
    def __init__(self, project_name):
        
        self.project_name=project_name
        
        # Set all system paths for projects and get github repos
        Project.check_all_github_repos() 
        
        Project.drives=Project.check_available_drives()
        Project.documents_path=Project.check_documents_path()        
        Project.main_drive=Project.documents_path.parts[0]       
        for drive in Project.drives:
            if drive not in Project.main_drive:
                Project.all_paths_for_this_system[drive]=os.path.join(drive,'\Projects')
                
        
        Project.dropbox_path=Project.check_dropbox_path()
           
        Project.all_paths_for_this_system['Github']=os.path.join(Project.documents_path,'Github')
        Project.all_paths_for_this_system['Documents']=os.path.join(Project.documents_path,'Projects')
        Project.all_paths_for_this_system['Dropbox']=os.path.join(Project.dropbox_path,'Projects')
        Project.long_all_paths_for_this_system={k:'\\\?\\' + v for k, v in Project.all_paths_for_this_system.items()}
        
        # Set specific projects paths        
        self.project_paths={}                        
        for key, value in  Project.long_all_paths_for_this_system.items():
            self.project_paths[key]=os.path.join(value, project_name)
        
        # Create specific project paths and  github repository    
        for value in  self.project_paths.values():
            if 'Github' not in value:
                self.create_project_folder_in_location(value)
            else:
               self.repo_object=self.solve_github_repo()


    @classmethod            
    def check_all_github_repos(self):
        for repo in Project.g.get_user().get_repos():
            Project.all_github_repos[repo.name]=repo


    @classmethod    
    def check_available_drives(self):   
         
        available_drives = ['%s:' % d for d in string.ascii_uppercase if os.path.exists('%s:' % d)]
        return available_drives
           
    
    @classmethod    
    def check_documents_path(self):   
        
        CSIDL_PERSONAL = 5       # My Documents
        SHGFP_TYPE_CURRENT = 0   # Get current, not default value
        
        buf= ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
        return Path(buf.value)
    
  
    @classmethod
    def check_dropbox_path(self):
        
        _system = system()

        if _system in ('Windows', 'cli'):           
            
            try:
                json_path = (Path(os.getenv('LOCALAPPDATA'))/'Dropbox'/'info.json').resolve()
            except FileNotFoundError:
                json_path = (Path(os.getenv('APPDATA'))/'Dropbox'/'info.json').resolve()
                
        elif _system in ('Linux', 'Darwin'):
            json_path = os.path.expanduser('~'
                                              '/.dropbox'
                                              '/info.json')
        else:
            raise RuntimeError('Unknown system={}'
                               .format(_system))
        if not os.path.exists(json_path):
            raise RuntimeError("Config path={} doesn't exists"
                               .format(json_path))  
                    
        with open(str(json_path)) as f:
            j = json.load(f)
            
        personal_dbox_path = Path(j['personal']['path'])
            
        return  personal_dbox_path
    
               
    def create_project_folder_in_location(self,location):         
        if not os.path.exists(location):
            os.makedirs(location)
        else:            
            print('project already there')
            
                          
    def read_repo_status(self): 
            try:
                _ = git.Repo(self.project_paths['Github']).git_dir
                print('Already a repo')
                return True
            except git.exc.InvalidGitRepositoryError:
                print('not repo')
                return False
        
               
    def solve_github_repo(self):       
         # no folder
        if not os.path.exists(self.project_paths['Github']):             
            if self.project_name in Project.all_github_repos.keys():
                # clone repo to new folder
                git.Git(Project.all_paths_for_this_system['Github']).clone(Project.all_github_repos[self.project_name].clone_url)
                repo_object=Repo(self.project_paths['Github'])
            else:  
                #make new repo in github                     
                self.github_repo= Project.u.create_repo(self.project_name)
                self.check_all_github_repos()
                git.Git(Project.all_paths_for_this_system['Github']).clone(Project.all_github_repos[self.project_name].clone_url)
                repo_object=Repo(self.project_paths['Github'])
        # folder no repo    
        elif self.read_repo_status():
            repo_object=Repo(self.project_paths['Github'])
            print('Already a repo, check manually')
            
        return repo_object
  
    def select_raw_primary_and_secondary_data(self):

        self.project_raw_data_path=select_values_gui(list(Project.all_paths_for_this_system.keys()),'RawData')
        self.project_secondary_data_path=select_values_gui(list(Project.all_paths_for_this_system.keys()),'SecondaryData')
        self.project_primary_data_path='Documents'

        
        
        
        
        
        
        