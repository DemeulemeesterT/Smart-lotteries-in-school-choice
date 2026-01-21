from .Data import *

def gale_shapley(MyData: Data):
    """
    Gale-Shapley algorithm.

    Parameters:
    - An instance from the Data class
    - Careful! Must contain strict priorities! (after tie-breaking!

    Returns:
    - A numpy array containing the assignment
    """
    n_stud = len(MyData.pref)
    n_schools = len(MyData.prior)
    pref = copy.deepcopy(MyData.pref) # We will gradually delete preferences from this 

    # Initialize data structures
    free_stud = list(range(n_stud))  # List of free students by index
    # Initialize temp_assigned with empty lists for each school
    temp_assigned = {school_index: [] for school_index in range(len(MyData.cap))} 

    while free_stud:
        # First we go through all students in 'free_stud' and remove them from the list
        while free_stud:
            i = free_stud[0]  # Get the first student
            free_stud.pop(0)  # Remove it
            # Assign free students to their most preferred school among remaining choices...
            # ... if preference list not empty yet
            if len(pref[i])>0:
                # Find index of that school
                index = MyData.ID_school.index(pref[i][0])
                temp_assigned[index].append(i) 
    
                # Remove that school from student i's preferences
                pref[i].pop(0)

        # Now each school j only keeps cap[j] most preferred students, and the others will be added to free_stud again
        for j in range(n_schools):
            if len(temp_assigned[j]) > MyData.cap[j]:
                # Dictionary containing priorities of the students who are temporarily assigned to school j
                prior_values = {stud_index: [] for stud_index in temp_assigned[j]}  
                for i in range(len(temp_assigned[j])):
                    # Find the position of student temp_assigned[j][i] in the priority list of school j
                    prior_values[temp_assigned[j][i]] = MyData.prior[j].index(MyData.ID_stud[temp_assigned[j][i]])

                # Sort the dictionary prior_values items by value
                sorted_prior_values = sorted(prior_values.items(), key=lambda item: item[1])

                # Remove the least preferred students who exceed capacity, and add them to free_students
                while len(temp_assigned[j]) > MyData.cap[j]:
                    # Add to free_stud
                    free_stud.append(sorted_prior_values[MyData.cap[j]][0])

                    # Remove from temp_assigned
                    temp_assigned[j].remove(sorted_prior_values[MyData.cap[j]][0])

                    # Remove from sorted_prior_values
                    sorted_prior_values.pop(MyData.cap[j])
    
    # Finally, transform the assignment in a numpy array where M[i][j] = 1 if student i is assigned to school j
    M = np.zeros(shape=(n_stud, n_schools))
    for j in range(n_schools):
        for k in range(len(temp_assigned[j])):
            M[temp_assigned[j][k]][j] = 1
    
    return M
    
