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
import psutil
import ctypes.wintypes
import os, string
from pathlib import Path
import json
from platform import system
import git
from git import Repo
from github import Github
import glob

class Project():
      
    def __init__(self, project_name, githubtoken_path, computer, platform):
        
        self.project_name=project_name
        print(f'Loading project {self.project_name}')
        Project.platform=platform
        Project.computer=computer
        
        # Set all system paths for projects and get github repos
        Project.check_all_github_repos() 
        
        Project.drives=Project.check_available_drives()
        Project.documents_path=Project.check_documents_path()        
        Project.main_drive=Project.documents_path.parts[0]       
        for drive in Project.drives:
            if drive not in Project.main_drive and 'H' not in drive:
                if Project.platform=='win32':
                    Project.all_paths_for_this_system[drive]=os.path.join(drive,'\Projects')
                elif Project.platform=='linux':
                    Project.all_paths_for_this_system[drive]=os.path.join(drive,'Projects')

        Project.dropbox_path=Project.check_dropbox_path()
           
        Project.all_paths_for_this_system['Github']=os.path.join(Project.documents_path,'Github')
        Project.all_paths_for_this_system['Documents']=os.path.join(Project.documents_path,'Projects')
        Project.all_paths_for_this_system['Dropbox']=os.path.join(Project.dropbox_path,'Projects')
        if Project.platform=='win32':
            Project.long_all_paths_for_this_system={k:'\\\?\\' + v for k, v in Project.all_paths_for_this_system.items()}
        elif Project.platform=='linux':
            Project.long_all_paths_for_this_system=Project.all_paths_for_this_system
        
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
        if Project.platform=='win32':
            available_drives = ['%s:' % d for d in string.ascii_uppercase if os.path.exists('%s:' % d)]
        elif Project.platform=='linux':
            available_drives0=['/']
            print('doing')
            mountpath=r'/media/sp3660'
            mountpath2=r'/mnt'
            available_drives1=glob.glob(mountpath+'/**')
            good_drives1=[]
            for drive in available_drives1:
                try:
                    os.listdir(drive)
                    good_drives1.append(drive)
                except:
                    print('mounting error')
            available_drives2=glob.glob(mountpath2+'/**')
            available_drives=available_drives0+good_drives1+[dr for dr in available_drives2 if '-8774-4ccf-8efb-700b593' in dr]

        return available_drives
           

             
    @classmethod    
    def check_documents_path(self):   
        if self.platform=='win32':
            CSIDL_PERSONAL = 5       # My Documents
            SHGFP_TYPE_CURRENT = 0   # Get current, not default value
            
            buf= ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
            doc_path=Path(buf.value)
          
        elif Project.platform=='linux':
            doc_path=Path(os.path.join(os.path.expanduser('~'),'Documents'))
            
            

        return doc_path
      
      
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
        # if self.platform=='win32':
        
            if not os.path.exists(location):
                os.makedirs(location)
            else:            
                print(f'project folder {location} already there')
                                      
    def read_repo_status(self): 
            try:
                _ = git.Repo(self.project_paths['Github']).git_dir
                print('Project is already a github repository')
                return True
            except git.exc.InvalidGitRepositoryError:
                print('Not a repo')
                return False
                       
    def solve_github_repo(self, project=False):       
         # no folder
        if not project: 
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
        else:
            repo_path=os.path.join(os.path.split(self.project_paths['Github'])[0],project)
            if not os.path.exists(repo_path):             
                if project in Project.all_github_repos.keys():
                    # clone repo to new folder
                    git.Git(Project.all_paths_for_this_system['Github']).clone(Project.all_github_repos[project].clone_url)                    
                    repo_object=Repo(repo_path)
            
        return repo_object
  

        
        
        
        
        
        