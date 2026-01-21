import numpy as np
import copy
from .Data import*
from .Assignment import *
from .DA_STB import generate_permutations_STB, generate_strict_prior_from_perturbation

from numpy.random import default_rng

def EADAM_STB(MyData: Data, n_iter: int, seed = 123456789, print_out = False):
    """
        EADAM with single tie-breaking

        Parameters:
        - MyData: An instance from the Data class
        - n_iter: number of tie-breakings sampled
        - print_out: boolean to control output on the screen

        Returns:
        - An instance of the Assignment class
    """

    permut = generate_permutations_STB(MyData, n_iter, seed, print_out)
    n_iter = len(permut) # Needed in case there were very few students and we consider all permutations


    # For each of the permutations, break ties in the preferences and run Gale-Shapley algorithm on them
    M_sum = np.zeros(shape=(MyData.n_stud, MyData.n_schools)) # Will contain the final random_assignment

    # Keep track of the generated matchings in a set. 
    # Only matchings that were not in the set yet will be added.
    M_set = set()
    w_set = {} # Contains the weights of the matchings in M_set. Each key of this set is a matching in M_set

    for p in tqdm(permut, desc='Generate EADAM_STB', unit = 'perturb', disable= not print_out):
        prior_new = generate_strict_prior_from_perturbation(MyData, p, print_out)
                
        # Compute DA matching for the new priorities after tie-breaking
        Data_new_prior = Data(MyData.n_stud, MyData.n_schools, MyData.pref, prior_new, MyData.cap, MyData.ID_stud, MyData.ID_school, MyData.file_name)
        
        # Compute EADAM for this tie-breaking
        consent = [1] * MyData.n_stud
        M_computed = EADAM(Data_new_prior, consent)

        M_set.add(tuple(map(tuple, M_computed))) # Add found matching to the set of matchings
            # You have to add it as a tuple of tuples, because otherwise Python cannot check whether it was already in the set.
            # If later you want to use it as a numpy array again, just use np.array(tuple_of_tuples)
       
        # Take note of the weight of this matching (depending on whether it was found before)
        key = tuple(map(tuple, M_computed))
        if key in w_set:
            w_set[key] = w_set[key] + 1/n_iter
        else:
            w_set[key] = 1/n_iter
        M_sum = M_sum + M_computed            
        
    M_sum = M_sum / n_iter

    # Create an instance of the Assignment class
    label = MyData.file_name + "_" + "EADAM_STB" + str(n_iter)
    A = Assignment(MyData, M_sum, M_set, w_set, label)

    return A


def EADAM(MyData: Data, consent: list, print_out = False):
    """
    Implements the simplified Efficiency Adjusted Deferred Acceptance Mechanism.

    This code is inspired by the accompanying code to Yuri, Faenza, and Xuan Zhang. "Legal Assignments and fast EADAM with consent via classical theory of stable matchings", OR 2022
    https://github.com/xz2569/FastEADAM 

    Args:
    - MyData: Data instance, contains student preferences and school priorities
    - consent (list of bool): A list where consent[i] is True if student i
                                consents, and False otherwise.

    Returns:
    - M: the matching, which is a binary numpy array of dimensions n_stud * n_schools
    """
    # Create mutable copies of the preference and rank structures
    temp_pref_index = copy.deepcopy(MyData.pref_index)
    temp_rank_prior = copy.deepcopy(MyData.rank_prior)

    iter = 0

    # Matching that will store final assignments
    M_final = np.zeros(shape=(MyData.n_stud, MyData.n_schools))

    while True:
        MyData

        iter = iter + 1

        # Step 1: Iteratively re-run Gale-Shapley
        M, ess_underdemand = EADAM_GS(temp_pref_index, temp_rank_prior, MyData.cap, print_out)

        if print_out:
            print("ITERATION ", iter)
            print(len(ess_underdemand), ' underdemanded schools:', ess_underdemand)
            print(M, "\n")

        # Step 2: If all schools are underdemanded, the algorithm terminates
        if len(ess_underdemand) == MyData.n_schools:
            break

        # Step 3: Modify preferences based on consent
        
        # Handle unassigned students
        # Identify unassigned agents
        unassigned_students = []
        for i in range(MyData.n_stud):
            if sum(M[i]) == 0:
                unassigned_students.append(i)
            
        for i in unassigned_students:
            if not consent[i]:
                # For each school this student had on their original list...
                for s in range(len(temp_pref_index[i])):
                    school = temp_pref_index[i][s]

                    student_rank_at_school = temp_rank_prior[school][i]

                    # Find students ranked lower by this school
                    worse_students = np.where(temp_rank_prior[s, :] > student_rank_at_school)[0]
                    
                    # The school must reject these worse-ranked students.
                    # We achieve this by removing the school from their preference lists.
                    for denied_student_idx in worse_students:
                        if s in temp_pref_index[denied_student_idx]:
                            temp_pref_index[denied_student_idx].remove(s)
            
            # The non-consenting unassigned student is removed from the market
            temp_pref_index[i] = []

        # Handle students assigned to underdemanded schools
        for school_idx in ess_underdemand:
            for i in range(MyData.n_stud):
                if M[i][school_idx] == 1: # If student i is assigned to underdemanded school school_idx
                    if not consent[i]:
                        # Get student's rank for their assigned school
                        rank_of_assigned_school = MyData.rank_pref[i, school_idx]

                        # Find schools the student preferred over their assignment
                        for pref_index in range(rank_of_assigned_school):
                            preferred_school = temp_pref_index[i][pref_index]
                            rank_student_at_pref_school = MyData.rank_prior[preferred_school, i]

                            # Find students ranked lower by this preferred school
                            worse_students = np.where(temp_rank_prior[preferred_school, :] > rank_student_at_pref_school)[0]

                            # This preferred school must reject these lower-ranked students
                            for denied_student_idx in worse_students:
                                if preferred_school in temp_pref_index[denied_student_idx]:
                                    temp_pref_index[denied_student_idx].remove(preferred_school)
                    
                    # Fix the assignment for this student (consenting or not)
                    # by removing all other schools from their list.
                    temp_pref_index[i] = [school_idx]
    
    # The final assignment from the last GS run is the result
    return M



def EADAM_GS(pref_index, rank_prior, cap, print_out = False):
    """
   This function is inspired by code to run gale-shapley algorithm, but keeps track of underdemanded schools
    Returns:
    - M (np.array): A numpy array where M[i][j] = 1 if student i is assigned to school j
    - ess_underdemand (list): A list of indices of essentially underdemanded schools.
    """
    
    n_stud = len(pref_index)
    n_schools = len(rank_prior)
    pref = copy.deepcopy(pref_index) # We will gradually delete preferences from this 

    # Initialize data structures
    free_stud = list(range(n_stud))  # List of free students by index
    # Initialize temp_assigned with empty lists for each school
    temp_assigned = {school_index: [] for school_index in range(len(cap))} 

    # Track all proposals made by students
    school_proposals = {s: set() for s in range(n_schools)}

    while free_stud:
        # First we go through all students in 'free_stud' and remove them from the list
        while free_stud:
            i = free_stud[0]  # Get the first student
            free_stud.pop(0)  # Remove it
            # Assign free students to their most preferred school among remaining choices...
            # ... if preference list not empty yet
            if len(pref[i])>0:
                # Find index of that school
                index = pref[i][0]
                temp_assigned[index].append(i) 
    
                # Remove that school from student i's preferences
                pref[i].pop(0)

                # CHANGE WITH RESPECT TO GALE SHAPLEY IMPLEMENTATION
                # Store the proposals to this school (used to identify essentially underdemanded schools later)
                school_proposals[index].add(i)
        
        # Now each school j only keeps cap[j] most preferred students, and the others will be added to free_stud again
        for j in range(n_schools):
            if len(temp_assigned[j]) > cap[j]:
                # Dictionary containing priorities of the students who are temporarily assigned to school j
                prior_values = {stud_index: [] for stud_index in temp_assigned[j]}  
                for i in range(len(temp_assigned[j])):
                    # Find the position of student temp_assigned[j][i] in the priority list of school j
                    prior_values[temp_assigned[j][i]] = rank_prior[j][temp_assigned[j][i]]

                # Sort the dictionary prior_values items by value
                sorted_prior_values = sorted(prior_values.items(), key=lambda item: item[1])

                # Remove the least preferred students who exceed capacity, and add them to free_students
                while len(temp_assigned[j]) > cap[j]:
                    # Add to free_stud
                    free_stud.append(sorted_prior_values[cap[j]][0])

                    # Remove from temp_assigned
                    temp_assigned[j].remove(sorted_prior_values[cap[j]][0])

                    # Remove from sorted_prior_values
                    sorted_prior_values.pop(cap[j])
    
    # Transform the assignment in a numpy array where M[i][j] = 1 if student i is assigned to school j
    M = np.zeros(shape=(n_stud, n_schools))
    for j in range(n_schools):
        for k in range(len(temp_assigned[j])):
            M[temp_assigned[j][k]][j] = 1


     # --- Part 2: Identify Essentially Underdemanded Schools 
    ess_underdemand = []
    
    # Get initial proposal counts for each school
    school_nproposals = {s: len(p) for s, p in school_proposals.items()}
    
    # Create a reverse mapping: student -> set of schools they proposed to
    student_to_proposals = {i: set() for i in range(n_stud)}
    for s, students in school_proposals.items():
        for i in students:
            student_to_proposals[i].add(s)

    # Iteratively find underdemanded schools
    while True:
        newly_found_count = 0
        for s in range(n_schools):
            # A school is underdemanded if its proposal count is not over capacity
            # and it hasn't already been identified.
            if school_nproposals[s] <= cap[s] and s not in ess_underdemand:
                ess_underdemand.append(s)
                newly_found_count += 1
                
                # Find all students who proposed to this newly underdemanded school
                proposing_students = list(school_proposals[s])

                # For each of these students, decrement the proposal counts for all *other*
                # schools they applied to. This "removes" them from other markets.
                for i in proposing_students:
                    for other_school_idx in student_to_proposals[i]:
                        if other_school_idx != s:
                            school_nproposals[other_school_idx] -= 1
                    # Ensure this student's proposals are not processed again
                    student_to_proposals[i].clear() 
        
        # If no new underdemanded schools were found in a full pass, stop.
        if newly_found_count == 0:
            break

    return M, ess_underdemand
    