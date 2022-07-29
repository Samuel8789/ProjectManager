# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:28:30 2021

@author: sp3660
"""


import importlib.util
import sys

from github import Github
from .ProjectsCLass import Project
from pprint import pprint
import os
import glob

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
        
        sys.path.insert(0,  self.all_projects_in_disk[project])
        sys.path.insert(0, self.all_projects_in_disk['ProjectManager'])

        if project=='LabNY':
            import ny_lab
            self.lab=ny_lab.RunNYLab( self.githubtoken_path, gui)
            return self.lab
        elif project=='AllenBrainObservatory':
            import allen
            self.allen_ob=allen.AllenBrainObservatory(self.githubtoken_path)
            return self.allen_ob
                   
    def check_project_git_status(self, project_object):
        repo=project_object.repo_object
        
        repo.is_dirty()  # check the dirty state
        repo.untracked_files  
        
        COMMITS_TO_PRINT = 5

        def print_repository_info(repo):
             print('Repository description: {}'.format(repo.description))
             print('Repository active branch is {}'.format(repo.active_branch))
         
             for remote in repo.remotes:
                 print('Remote named "{}" with URL "{}"'.format(remote, remote.url))
         
             print('Last commit for repository is {}.'.format(str(repo.head.commit.hexsha)))
                 
        
        def print_commit_data(commit):
            print('-----')
            print(str(commit.hexsha))
            print("\"{}\" by {} ({})".format(commit.summary, commit.author.name, commit.author.email))
            print(str(commit.authored_datetime))
            print(str("count: {} and size: {}".format(commit.count(), commit.size)))
            
        # check that the repository loaded correctly
        if not repo.bare:
            print('Repo at {} successfully loaded.'.format(repo.working_tree_dir))
            print_repository_info(repo)
        
            # create list of commits then print some of them to stdout
            commits = list(repo.iter_commits('master'))[:COMMITS_TO_PRINT]
            for commit in commits:
                print_commit_data(commit)
                pass
        
        else:
            print('Could not load repository at {} :'.format(repo.working_tree_dir))
            
        git = project_object.repo_object.git
        pprint(git.status())
    
    def stage_commit_and_push(self, project_object):
        git = project_object.repo_object.git
        pprint(git.status())
        git.add('--all')
        git.commit ('-m' ,"general auto commit")
        git.push('origin' ,'master')
        pprint(git.status())
        
    def pull_from_github(self, project_object):
        git = project_object.repo_object.git
        pprint(git.status())
        git.fetch()
        git.log('HEAD..origin/master')
        git.log('-p','HEAD..origin/master')
        git.diff('HEAD...origin/master')
        git.pull('origin' ,'master')
        pprint(git.status())








    
        
