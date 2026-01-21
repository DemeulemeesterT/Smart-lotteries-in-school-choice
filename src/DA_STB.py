from .Assignment import *
from .GaleShapley import gale_shapley
from .SICs import SIC
from.ErdilErgin import *

from numpy.random import default_rng
import time
import math
import itertools

def DA_STB(MyData: Data, n_iter: int, DA_impl: str, bool_SIC: bool, seed = 123456789, print_out = False):
    """
    Deferred Acceptance with single tie-breaking (my implementation)

    Parameters:
    - MyData: An instance from the Data class
    - n_iter: number of tie-breakings sampled
    - DA_impl: string that determines which implementation is used 
        - 'GS': Gale-Shapley as implemented by me
        - 'EE': implementation by Erdil & Ergin (2008)
    - bool_SIC: boolean that determines whether or not we immediately find SICs as well
    - print_out: boolean to control output on the screen

    Returns:
    - An instance of the Assignment class
    """

    DA_impl_list = ['GS', 'EE']
    if DA_impl not in DA_impl_list:
           raise ValueError(f"Invalid value: '{DA_impl}'. Allowed values are: {DA_impl_list}")

    permut = generate_permutations_STB(MyData, n_iter, seed, print_out)
    n_iter = len(permut) # Needed in case there were very few students and we consider all permutations

    # For each of the permutations, break ties in the preferences and run Gale-Shapley algorithm on them
    M_sum = np.zeros(shape=(MyData.n_stud, MyData.n_schools)) # Will contain the final random_assignment

    # Keep track of the generated matchings in a set. 
    # Only matchings that were not in the set yet will be added.
    M_set = set()
    w_set = {} # Contains the weights of the matchings in M_set. Each key of this set is a matching in M_set

    counter = 0
    for p in tqdm(permut, desc='Generate DA_STB', unit = 'perturb', disable= not print_out):
        prior_new = generate_strict_prior_from_perturbation(MyData, p, print_out)
                
        # Compute DA matching for the new priorities after tie-breaking
        Data_new_prior = Data(MyData.n_stud, MyData.n_schools, MyData.pref, prior_new, MyData.cap, MyData.ID_stud, MyData.ID_school, MyData.file_name)
        if DA_impl == "GS":
            M_computed = gale_shapley(Data_new_prior)
            
        elif DA_impl == "EE":
            # Change format of preferences and priorities
            N = transform_pref_us_to_EE(Data_new_prior)
            A = transform_prior_us_to_EE(Data_new_prior)
            Q = MyData.cap
            result = DA_Erdil_ergin(N, A, Q, False)

            M_computed = transform_M_EE_to_us(Data_new_prior, result['stable_all'], print_out)
        
        # Compute SICs, if desired
        if bool_SIC:
            M_computed = SIC(Data_new_prior, M_computed, False)
        

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
        counter = counter + 1            
        
    M_sum = M_sum / n_iter

    # Create an instance of the Assignment class
    label = MyData.file_name + "_" + "DA_STB" + str(n_iter)
    A = Assignment(MyData, M_sum, M_set, w_set, label)

    return A

def generate_permutations_STB(MyData: Data, n_iter: int, seed = 123456789, print_out = False):
    """
        Will create 'n_iter' permutations of the students

        Returns a list 'permut' containing the permutations
    """
    
    if seed != 123456789:
        # Use seed in argument
        rng = default_rng(seed)
    else:
        # Generate random seed 
        # Create a seed based on the current time
        seed = int(time.time() * 1000) % (2**32)  # Modulo 2^32 to ensure it's a valid seed

    np.random.seed(seed)

    # First, check how many tie-breaking rules would be needed in total
    # Look at total number of students who are included in ties
    students_in_ties = set()
    for j in range(MyData.n_schools):
        for k in range(len(MyData.prior[j])):
            if isinstance(MyData.prior[j][k], tuple): # When more than a single student in this element
                for l in range(len(MyData.prior[j][k])):
                    # students_in_ties.add(MyData.ID_stud.index(MyData.prior[j][k][l])) # We add the index of this student, not its name
                    students_in_ties.add(MyData.prior[j][k][l])

    students_in_ties = list(students_in_ties) # Convert the set to a list, allows us to access k-th element
    
    # The total number of needed tie-breaking rules is m!, where m = |student_in_ties|
    n_STB = math.factorial(len(students_in_ties))

    # We only need to perturb the students who appear in ties:
    
    if n_STB < n_iter:
        n_iter = n_STB
        # Enumerate all relevant permutations
        permut = list(itertools.permutations(students_in_ties))
    else:
        permut = set() # We first create a set, to ensure that all found permutations are unique. Later, convert to list
        # Sample n_iter out of all n_STB relevant permutations
        while len(permut) < n_iter:
            np.random.shuffle(students_in_ties)  # Shuffle in place
            permut.add(tuple(students_in_ties))
        permut = list(permut)
    
    if print_out:
        print(f"Students in ties: {len(students_in_ties)}")
        if MyData.n_stud <=100:
            print(f"Tie-breaking rules needed: {n_STB}")
        print(f"Tie-breaking rules sampled: {n_iter}")
        # print(f"permut: {permut}")

    return permut

def generate_strict_prior_from_perturbation(MyData: Data, permut: tuple, print_out = False):
    prior_new = [] 
    for j in range(len(MyData.prior)):
        # Just add priorities if no ties:
        if len(MyData.prior[j]) == MyData.n_stud:
            prior_new.append(MyData.prior[j])
        else:
            prior_array = []
            for k in range(len(MyData.prior[j])):
                if isinstance(MyData.prior[j][k], tuple): # set of students who have same priorities
                    # Reorder the students based on the permuation
                    reordered_prior = list(sorted(MyData.prior[j][k], key=lambda x: permut.index(x)))

                    # Add to prior_array
                    for l in range(len(MyData.prior[j][k])):
                        prior_array.append(reordered_prior[l])
                else:
                    prior_array.append(MyData.prior[j][k])                        
            prior_new.append(prior_array)
    return prior_new