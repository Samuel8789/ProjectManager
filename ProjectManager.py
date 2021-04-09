# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:28:30 2021

@author: sp3660
"""


import sys

sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/ProjectManager')

from ProjectsCLass import Project


class ProjectManager(Project):    
    def __init__(self):
        Project.__init__(self, 'ProjectManager')
        
        self.main_directory='\\\?\\' +self.project_paths['Documents']  
