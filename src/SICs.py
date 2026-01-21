from .Assignment import *
from.ErdilErgin import *

def SIC_all_matchings(MyData: Data, A: Assignment, print_out = False):
    """
    Will compute the SICs for all matchings in the decomposition, 
    will update M_set,
    and will compute the new assignment probabilities
    """
    if print_out: 
        print("STATISTICS BEFORE:")
        A.statistics(print_out)

    # Initialize variables used to define new assignment object
    M_sum = np.zeros(shape=(MyData.n_stud, MyData.n_schools)) # Will contain the final random_assignment
    M_set = set()
    w_set = {}

    # Go through all matchings
    counter = 0
    for M in tqdm(A.M_set, desc='Compute SICs for matchings', unit = 'matching', disable= not print_out):
        # Weight of this matching in the assignment:
        w_original = A.w_set[M]

        # Find SICs
        M_out = SIC(MyData, M, False)
        

        M_set.add(tuple(map(tuple,M_out))) # This is needed because numpy arrays cannot be used as keys in a dictionary
        key2 = tuple(map(tuple, M_out))
        w_set[key2] = w_set.get(key2, 0.0) + w_original
    
        # Compute the new assignment
        M_sum = M_sum + M_out * w_original

        counter +=1

    # Create a new Assignment object
    label = A.label + "_SIC"
    A_out = Assignment(MyData, M_sum, M_set, w_set, label)

    if print_out:
        print("STATISTICS AFTER:")
        A_out.statistics(print_out)

    return A_out


def SIC(MyData: Data, M: np.ndarray, print_out = False):
    """
    Version of the function that takes our format for matchings and data as inputs
    """
    # Change format of preferences and priorities
    N = transform_pref_us_to_EE(MyData)
    A = transform_prior_us_to_EE(MyData)
    Q = MyData.cap
    M_EE = transform_M_us_to_EE(MyData, M, print_out)

    # Store for each student to which preference they are assigned
    assigned_pref = [MyData.n_schools] * len(N)
    for i in range(MyData.n_stud):
        for k in range(len(MyData.pref_index[i])): # Go through all pref
            if M[i][MyData.pref_index[i][k]] == 1: # If assigned to object on k-th spot in preferences of student i
                assigned_pref[i] = k 
                k = len(MyData.pref_index[i]) # Finish for-loop

    if print_out:
        print("Assigned pref (before):", assigned_pref)       

    result = SIC_EE(N, A, Q, M_EE, assigned_pref, print_out)

    if print_out:
        print("Assigned pref (after):", result['optimal_all'])
    
    """
    Contains:
        'optimal_all': the final matching (in their format)
        'iterations'
        'total_moves'
        'total_cycles'
        'improved_students'
    """


    M_out = transform_M_EE_to_us(MyData, result['optimal_all'], print_out)

    if print_out:
        print("Matching after")
        print(np.array(M_out))

    return M_out


def SIC_all_matchings_with_mistake_EE(MyData: Data, A: Assignment, print_out = False):
    """
    Will compute the SICs for all matchings in the decomposition, 
    will update M_set,
    and will compute the new assignment probabilities
    Contains the bug that was initially made in code by Erdil & Ergin
    """
    if print_out: 
        print("STATISTICS BEFORE:")
        A.statistics(print_out)

    # Initialize variables used to define new assignment object
    M_sum = np.zeros(shape=(MyData.n_stud, MyData.n_schools)) # Will contain the final random_assignment
    M_set = set()
    w_set = {}

    # Go through all matchings
    counter = 0
    for M in tqdm(A.M_set, desc='Compute SICs for matchings', unit = 'matching', disable= not print_out):
        # Weight of this matching in the assignment:
        w_original = A.w_set[M]

        # Find SICs
        M_out = SIC_with_mistake_EE(MyData, M, print_out)
        

        M_set.add(tuple(map(tuple,M_out))) # This is needed because numpy arrays cannot be used as keys in a dictionary
        key2 = tuple(map(tuple, M_out))
        w_set[key2] = w_set.get(key2, 0.0) + w_original
    
        # Compute the new assignment
        M_sum = M_sum + M_out * w_original

        counter +=1

    # Create a new Assignment object
    label = A.label + "_SIC"
    A_out = Assignment(MyData, M_sum, M_set, w_set, label)

    if print_out:
        print("STATISTICS AFTER:")
        A_out.statistics(print_out)

    return A_out

def SIC_with_mistake_EE(MyData: Data, M: np.ndarray, print_out = False):
    """
    Version of the function that takes our format for matchings and data as inputs
    """
    # Change format of preferences and priorities
    N = transform_pref_us_to_EE(MyData)
    A = transform_prior_us_to_EE(MyData)
    Q = MyData.cap
    M_EE = transform_M_us_to_EE(MyData, M, print_out)

    # Store for each student to which preference they are assigned
    assigned_pref = [MyData.n_schools] * len(N)
    for i in range(MyData.n_stud):
        for k in range(len(MyData.pref_index[i])): # Go through all pref
            if M[i][MyData.pref_index[i][k]] == 1: # If assigned to object on k-th spot in preferences of student i
                assigned_pref[i] = k 
                k = len(MyData.pref_index[i]) # Finish for-loop

    if print_out:
        print("Assigned pref (before):", assigned_pref)

    result = SIC_EE_with_mistake_EE(N, A, Q, M_EE, assigned_pref, print_out)

    if print_out:
        print("Assigned pref (after):", result['optimal_all'])
        print('\n N:', N, '\n',
              'A:', A,'\n',
              'Q:', Q,'\n',
              'allocation:', M_EE,'\n',
              'pro_off:', assigned_pref, '\n',)
    
    """
    Contains:
        'optimal_all': the final matching (in their format)
        'iterations'
        'total_moves'
        'total_cycles'
        'improved_students'
    """


    M_out = transform_M_EE_to_us(MyData, result['optimal_all'], print_out)

    if print_out:
        print("Matching after")
        print(np.array(M_out))

    return M_out


    