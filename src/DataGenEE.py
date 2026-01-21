## Written by Ozgur Sumer
## Based on code from Erdil & Ergin (2008, AER) paper: https://www.openicpsr.org/openicpsr/project/113247/version/V1/view

from src.ErdilErgin import *
import random
import pylab as pl
import math
import time

VERBOSE = False
DISPLAY = False

"""
The function DataGenEE() generates data according to the method by Erdil & Ergin, and outputs it in our format
IMPORTANTLY, we also added a seed to make the data generation process more reproducible

In this file, the following notation is used (I added examples for two schools and five students)
- A: school priorities ([{0: 0, 1: 0, 2: 0, 3: 1, 4: 1}, {0: 1, 1: 1, 2: 1, 3: 0, 4: 0}])
- Q: school capacities ([3.5, 3.5])
- N: student preferences ([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]])
- U: utilities of the students for the schools ([[1.8356485489646472, 0.006903626328407353], [0.3004739990745144, -0.7063682762519852], [0.13916835917220605, 0.8730710020259577], [-0.6693792234085418, 0.954527884625698], [0.05757674029889143, 0.017677713247009286]])
- s2d_map: To which school's catchment area the students belong ( [0, 0, 0, 1, 1])
- sch: the coordinates of the schools ([(0.5816448398107481, 0.5771843898098422), (0.29906698846762736, 0.4300487331366828)])
- stu: the coordinates of the students ([(0.7961687378795701, 0.26210413466032867), (0.8137843656909839, 0.9722910859531648), (0.27632817894041295, 0.9077954113742779), (0.09555333266756993, 0.6246217950264992), (0.18760859762095383, 0.3756565134396317)])
 
 When in doubt, run the test() function, which will plot the catchment areas, and print the student preferences
"""

def DataGenEE(n_students: int, n_schools: int, alpha: float, beta: float, pref_list_length: int, print_data=False, seed=123456789):
    """
    Generate data according to Erdil & Ergin (2008).

    Parameters:
    - n_students: Number of students.
    - n_schools: Number of schools.
    - alpha, beta: Parameters to determine student preferences
    - pref_list_length: length of the students preferences
    - print_data: Whether to print the generated data.
    - seed: Random seed for reproducibility.

    Returns:
    - An object of the class Data
    """
     
    if seed == 123456789:
        # Generate random seed 
        # Create a seed based on the current time
        seed = int(time.time() * 1000) % (2**32)  # Modulo 2^32 to ensure it's a valid seed

    random.seed(seed) 

    # Generate data output in Erdil & Ergin format:
    param = {}
    param['alpha'] = alpha
    param['beta'] = beta

    # Generate data
    result = cor_gen(n_students, n_schools, pref_list_length, param)
    #print(result['student_utilities'])
    #print(result['N'])
    #print(result['A'])
    N = result['N'] #pref
    A = result['A'] #prior
    Q = result['Q'] #cap

    # Transform format
    # Preferences & capacities: okay
    A = transform_prior_EE_to_us(A)
    #print(A)

    stud =  list(range(n_students))
    schools = list(range(n_schools))

    name = str(n_students) + '_' + str(n_schools) + '_' + str(alpha) + '_' + str(beta) + '_' + str(seed)

    MyData = Data(n_students, n_schools, N, A, Q, stud, schools, name)

    return MyData

def cor_gen(n, a, sq, param):
    alpha = param.get('alpha', 0)
    beta = param.get('beta', 0)
    A,Q, extra = district_priorities(n, a)
    N, U = correlated_student_preferences(n, a, sq, extra + [alpha, beta])
    return {'N' : N, 'A' : A, 'Q' : Q, 'student_utilities' : U}   

def gen_grid(n ,a):
    school_grid = []
    for school in range(a):
        school_grid.append((random.random(), random.random()))

    student_grid = []
    for student in range(n):
        student_grid.append((random.random(), random.random()))

    return school_grid, student_grid

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def student_to_district_map(school_grid, student_grid):
    s2d_map = []
    for st_x, st_y in student_grid:
        min_school_index = 0
        min_school_distance = 999999
        for school_index in range(len(school_grid)):
            sch_x, sch_y = school_grid[school_index]
            if distance(st_x, st_y, sch_x, sch_y) < min_school_distance :
                min_school_index = school_index
                min_school_distance = distance(st_x, st_y, sch_x, sch_y)
        s2d_map.append(min_school_index)
            
    return s2d_map

def district_priorities(n, a, grid_fun = gen_grid):
    A = []

    capacity = n/a + 1
    if n % a == 0:
        capacity = n/a

    Q = [capacity] * a
    Q = [int(q) for q in Q] # Originally this wasn't integer, but I modified this

    school_grid, student_grid = grid_fun(n, a)
    s2d_map = student_to_district_map(school_grid, student_grid)

    for school in range(a):
        rank = {}
        for student in range(n):
            if s2d_map[student] == school :
                rank[student] = 0
            else :
                rank[student] = 1
        A.append(rank)

    if VERBOSE:
        print('School Priorities')
        printS(A)
        print('Capacities:', Q)
        

    return A, Q, [s2d_map, school_grid, student_grid]


def XY_normal_rv(n, a):
    Y = []
    for school in range(a):
        Y.append(random.normalvariate(0,1))

    X = []
    for student in range(n):
        stu_rv = []
        for school in range(a):
            stu_rv.append(random.normalvariate(0,1))
        X.append(stu_rv)

    return X, Y
    

def correlated_student_preferences(n, a, sq, extra, XY_gen_fun = XY_normal_rv):
    N = []
    s2d_map, school_grid, student_grid, alpha, beta = extra

    X, Y = XY_gen_fun(n, a)

    U = []
    for student in range(n):
        row = []
        for school in range(a):
            utility = X[student][school]
##            print student, school, utility,
            utility = alpha * Y[school] + (1 - alpha) * utility
##            print '--a-->', utility,
            utility = - beta * distance(school_grid[school][0], \
                                        school_grid[school][1], \
                                        student_grid[student][0], \
                                        student_grid[student][1]) + \
                      (1 - beta) * utility
##            print '--b-->', utility
            row.append(utility)
        U.append(row)

    for student in range(n):
        utilities = pl.array(U[student])
        ranking = list(utilities.argsort())
        ranking.reverse()
        N.append(ranking[:sq])

    if VERBOSE:
        print('Student Utilities over Schools')
        printS(U)
        print()
        print('Student Preferences')
        printS(N)
        print()
        
    return N, U

def utility_stat(param):
    all_before = param['stable_all']
    all_after = param['optimal_all']
    U = param['student_utilities']

    def total_utility(all):
        total = 0
        count = 0
        for school, school_population in zip(list(range(len(all))), all):
            for student in school_population :
                count += 1
                total += U[student][school]
        return count, total

    count, util_before = total_utility(all_before)
    count, util_after = total_utility(all_after)

    return {
        'per_capita_util_imp' : float(util_after - util_before)/count ,\
        }


def test(alpha_in, beta_in):
    n = 5
    a = 2
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    A, Q, extra = district_priorities(n, a)
    s2d_map, sch, stu = extra
    print('A', A)
    print('Q', Q)
    print('s2d_map', s2d_map)
    print('sch' ,sch)
    print('stu', stu)
    alpha = alpha_in
    beta = beta_in
    extra = [s2d_map, sch, stu, alpha, beta]
    N, U = correlated_student_preferences(n, a, 5, extra)
    print('N', N)
    print('U', U)

    for school in range(a):
        color = colors[school % 7]
        pl.plot([sch[school][0]], [sch[school][1]], color + 'o')
        student_indices = [el for el in range(len(stu)) if s2d_map[el] == school]
        pl.plot([stu[i][0] for i in student_indices], \
                [stu[i][1] for i in student_indices], color + '.')
        



def printS(l):
    if type(l) == type({}):
        print('{')
    elif type(l) == type([]):
        print('[')
    else :
        print(l)
        return
    for el in l:
        print(('\t', el))
    if type(l) == type({}):
        print('}')
    elif type(l) == type([]):
        print(']')
        
    
