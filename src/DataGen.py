from .Data import *

from numpy.random import default_rng
import time

class DataGenParam:
    # Parameters to used for data generation, see https://github.com/DemeulemeesterT/GOSMI
    
    def __init__(self, capacity_ratio=1.2, corr_cap_pop=0.21, mean_pref=2.42, sigma_pref=1.05, 
                 CV_cap=0.8, CV_pop=0.6, delta_1=0.14, delta_2=0.009, pop_percentage=0.10):
        self.capacity_ratio = capacity_ratio
        self.corr_cap_pop = corr_cap_pop
        self.mean_pref = mean_pref
        self.sigma_pref = sigma_pref
        self.CV_cap = CV_cap
        self.CV_pop = CV_pop
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.pop_percentage = pop_percentage

def generate_data(n_students: int, n_schools: int, parameters: DataGenParam, name: str, print_data=False, seed=123456789):
    """
    Generate data. The preferences and capacities are based on: https://github.com/DemeulemeesterT/GOSMI

    Parameters:
    - n_students: Number of students.
    - n_schools: Number of schools.
    - parameters: An instance of DataGenParam with generation parameters.
    - print_data: Whether to print the generated data.
    - seed: Random seed for reproducibility.

    Returns:
    - An object of the class Data
    """
    if seed == 123456789:
        # Generate random seed 
        # Create a seed based on the current time
        seed = int(time.time() * 1000) % (2**32)  # Modulo 2^32 to ensure it's a valid seed

    rng = default_rng(seed)  # Always define rng


    # Initialize arrays
    students = list(range(n_students))
    schools = list(range(n_schools))

    # Generate capacities and popularity
    capacity_total = int(round(parameters.capacity_ratio * n_students))
    capacity_aid = rng.normal(0, 1, n_schools)
    popularity_aid = rng.normal(0, 1, n_schools)
    
    # Cholesky decomposition for correlated random variables
    covar = np.array([
        [1, parameters.corr_cap_pop],
        [parameters.corr_cap_pop, 1]
    ])
    G = np.linalg.cholesky(covar).T
    correlated = G @ np.vstack((capacity_aid, popularity_aid))
    capacity_aid, popularity_aid = correlated[0], correlated[1]

    # Rescale capacities
    mean_capacity_aid = np.mean(capacity_aid)
    capacities = np.round((capacity_total / n_schools) * 
                          (1 + parameters.CV_cap * (capacity_aid - mean_capacity_aid)))
    capacities = np.clip(capacities, 1, None).astype(int)  # Ensure no capacity < 1
    scale_factor = capacity_total / np.sum(capacities)
    capacities = np.round(capacities * scale_factor).astype(int)

    # Generate preference lengths
    pref_lengths = rng.normal(parameters.mean_pref, parameters.sigma_pref, n_students)
    pref_lengths = np.clip(pref_lengths, 1, n_schools).astype(int)

    # Determine popularity thresholds
    mean_pop_ratio = np.sum(pref_lengths) / np.sum(capacities)
    mean_popularity_aid = np.mean(popularity_aid)
    popularity_aid = mean_pop_ratio * (1 + parameters.CV_pop * (popularity_aid - mean_popularity_aid))
    popularity_aid = np.clip(popularity_aid, 0.2, None)  # Ensure no popularity < 0.2
    requests = capacities * popularity_aid
    popularity_aid *= np.sum(pref_lengths) / np.sum(requests)

    # Define popular schools
    sorted_indices = np.argsort(-popularity_aid)
    popularity_threshold = popularity_aid[sorted_indices[int(parameters.pop_percentage * n_schools)]]
    popular = popularity_aid > popularity_threshold

    # Generate preferences for each student
    preferences = []
    for i in range(n_students):
        student_pref = []
        available_schools = list(range(n_schools))
        for _ in range(pref_lengths[i]):
            weights = popularity_aid[available_schools]
            weights /= np.sum(weights)
            choice = rng.choice(available_schools, p=weights)
            student_pref.append(choice) 
            available_schools.remove(choice)
        preferences.append(student_pref)

    # Generate random school priorities:
    # For now, just simply
        # Randomly order students for each school
        # Divide into three indifference groups for each school

    priorities = []
    for j in range(n_schools):
        permutation = rng.permutation(students)

        # Split the list into three roughly equal groups
        group_size = len(permutation) // 3
        group1 = permutation[:group_size]
        group2 = permutation[group_size:2 * group_size]
        group3 = permutation[2 * group_size:]

        if len(group1) == 1:
            tuple_group1 = group1
        else:
            tuple_group1 = tuple(group1)

        if len(group2) == 1:
            tuple_group2 = group2
        else:
            tuple_group2 = tuple(group2)

        if len(group3) == 1:
            tuple_group3 = group3
        else:
            tuple_group3 = tuple(group3)

        priorities.append([tuple_group1, tuple_group2, tuple_group3])

        #priorities.append([tuple(group1), tuple(group2), tuple(group3)])
    
    # Optionally print data
    if print_data:
        print(f"Generated data with {n_students} students and {n_schools} schools.")
        print(f"Preferences: {preferences}")
        print(f"Priorities: {priorities}")
        print(f"Capacities: {list(capacities)}")
        # print(f"Popularity Ratios: {list(popularity_aid)}")
        print(f"Students: {students}")
        print(f"Schools: {schools}")
    
    MyData = Data(n_students, n_schools, preferences, priorities, capacities, students, schools, name)
    
    # Return results
    return MyData