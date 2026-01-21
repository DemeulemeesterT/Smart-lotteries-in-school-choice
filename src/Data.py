import numpy as np
import copy # To make deep copies
import os
from tqdm import tqdm # To show progress bar
    # If you import from tqdm.auto, it will import notebook version of tqdm automatically when in notebook
import pandas as pd
import csv
import random
import pickle # To export data


# Comment out if pulp and gurobipy are already installed
#! pip install pulp
#! pip install gurobipy

class Data:
    """
    Class Data:
    A representation of students and schools for allocation problems. 

    Arguments for initialization:
    - n_stud (int): Number of students.
    - n_schools (int): Number of schools.
    - pref (list): List of student preferences for schools (by IDs).
    - prior (list): List of school priorities for students (by IDs or tuples of IDs for grouped priorities).
    - cap (list): Capacities of each school.
    - ID_stud (list): List of student identifiers.
    - ID_school (list): List of school identifiers.
    - file_name (str): Name of the data file for reference.
    """

    # Define the initialization of an object from this class
    def __init__(self, n_stud: int, n_schools: int, pref: list, prior: list, cap:list, ID_stud:list, ID_school:list, file_name:str):
        self.n_stud = n_stud
        self.n_schools = n_schools
        self.pref = copy.deepcopy(pref)
        self.prior = copy.deepcopy(prior)
        self.cap = copy.deepcopy(cap)
        self.ID_stud = copy.deepcopy(ID_stud)
        self.ID_school = copy.deepcopy(ID_school)
        self.file_name = file_name   
        self.file_name = os.path.basename(self.file_name)  
        if '/' in self.file_name:
            # If name contains slash, problems with directories. Replace by _
            self.file_name = self.file_name.replace('/', '_')
        
        if '.' in self.file_name:
            self.file_name = self.file_name.replace('.', '_')


        # Create alternative copies of pref and prior in which the elements are no longer strings, 
        # but the indices of the corresponding elements in the ID vectors
        self.pref_index = [[self.ID_school.index(school) for school in student_pref] for student_pref in self.pref]

        # Now create two matrices containing the position of the schools in the preferences, and of the students in the priorities
        # Initialize the rank matrix with NaN
        self.rank_pref = np.full((self.n_stud, self.n_schools), np.nan)

        self.prior_index = []
        for school_prior in self.prior:
            transformed_school_prior = []
            for student_group in school_prior:
                if isinstance(student_group, tuple):
                    transformed_school_prior.append(tuple(ID_stud.index(student) for student in student_group))
                else:
                    transformed_school_prior.append(ID_stud.index(student_group))
            self.prior_index.append(transformed_school_prior)

        
        # Populate the rank matrix
        for i, student_pref in enumerate(self.pref):
            for rank_position, school_id in enumerate(student_pref):
                if school_id in self.ID_school:
                    school_index = self.ID_school.index(school_id)
                    self.rank_pref[i][school_index] = rank_position

        # Initialize the rank_prior matrix with NaN
        self.rank_prior = np.full((self.n_schools, self.n_stud), np.nan)
        
        # Populate the rank_prior matrix
        for j, school_prior in enumerate(self.prior):
            for rank_position, student_id in enumerate(school_prior):
                # Handle tuple (grouped students) by expanding
                if isinstance(student_id, tuple):
                    for grouped_student in student_id:
                        if grouped_student in self.ID_stud:
                            student_index = self.ID_stud.index(grouped_student)
                            self.rank_prior[j][student_index] = rank_position + 1  # Positions are 1-based
                elif student_id in self.ID_stud:
                    student_index = self.ID_stud.index(student_id)
                    self.rank_prior[j][student_index] = rank_position + 1  # Positions are 1-based
    
    # Choose what is being shown for the command 'print(MyData)', where 'MyData' is an instance of the class 'Data'
    def __str__(self):
        s ="The data instance has the following properties: \n"
        s += f"\n\t{self.n_stud} students.\n\t{self.n_schools} schools. \n\n \tPREFERENCES:\n"
        for i in range(0,self.n_stud):
            s+= f"\t{self.ID_stud[i]}\t"
            for j in range(0, len(self.pref[i])):
                s+=f"{self.pref[i][j]} "
            s +="\n"

        s += f"\n\n \tCAPACITIES & PRIORITIES:\n"
        for i in range(0,self.n_schools):
            s+= f"\t{self.ID_school[i]}\t"
            s+= f"{self.cap[i]}\t"
            for j in range(0, len(self.prior[i])):
                #if len(self.prior[i][j]) >= 2:
                if isinstance(self.prior[i][j], tuple):
                    s+=f"{{"
                    for k in range(0, len(self.prior[i][j])):
                        s+=f"{self.prior[i][j][k]}"
                        if k < len(self.prior[i][j]) - 1:
                            s+= f" "
                    s+=f"}} "
                else:
                    s+=f"{self.prior[i][j]} "
            s +="\n"
        return s