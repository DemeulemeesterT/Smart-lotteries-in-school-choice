from .Data import *
from random import random
from copy import deepcopy



# Many functions in this code are based on the code that was published by Erdil & Ergin:
# https://www.openicpsr.org/openicpsr/project/113247/version/V1/view?path=/openicpsr/113247/fcr:versions/V1&type=project 

# We chose to:
    # transform the format in which we store preferences, priorities, and matchings to their format
    # run their code for the algorithms (such as Stable Improvement Cycles)
    # transform the returned matching to our format again

# OVERVIEW
    # PREFERENCES:
        # WE: MyData.pref: can possibly contain strings
        # EE: can't contain strings, index starts from 0
            # --> Use MyData.pref_index

    # PRIORITIES:
        # WE: MyData.prior is list, but can contain tuples to indicate ties
            # i.e., [(4, 5), 0, (1, 2, 3)]
        # EE: Dictionary, where for each student, the element contains the rank (with possible ties)
            # i.e., W = [
            #            {0:1, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:2, 14:2, 15:1},
            #            {0:1, 1:2, 2:1, 3:2, 4:1, 5:1, 6:1, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:2, 14:1, 15:1},
            #            {0:2, 1:1, 2:1, 3:2, 4:1, 5:1, 6:1, 7:2, 8:2, 9:2, 10:2, 11:2, 12:1, 13:2, 14:2, 15:1},
            #            {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:2, 14:2, 15:1}
            #]

    # CAPACITIES:
        # Same in both codes

    # MATCHING:
        # WE: matching is numpy array where M[i][j] = 1 if student i is assigned to school j, and 0 otherwise
            # i.e., [[0. 0. 1. 0.]
            #        [1. 0. 0. 0.]
            #        [0. 1. 0. 0.]
            #        [0. 0. 0. 1.]]
        # EE: list where M[j] contains the set of students that are assigned to school j
            # i.e., [{1}, {2}, {0}, {3}]

def transform_pref_us_to_EE(MyData: Data, print_out = False):
    return MyData.pref_index

# No need to transform from EE to us, we simply set pref to be identical to their format (which is what we called pref_index)    

def transform_prior_us_to_EE(MyData: Data, print_out = False):
    W = []
    for j in range(MyData.n_schools):
        student_to_group = {}
    
        for group_index, group in enumerate(MyData.prior_index[j]):
            if isinstance(group, tuple):
                # Handle tuple of students - all belong to same priority group
                for student in group:
                    student_to_group[student] = group_index
            else:
                # Handle individual student
                student_to_group[group] = group_index
        
        W.append(student_to_group)
    return W

def transform_prior_EE_to_us(prior_in: list, print_out = False):
    prior = []
    for j in range(len(prior_in)):
        # Group students by their priority group number
        groups = {}
        for student, group_num in prior_in[j].items():
            if group_num not in groups:
                groups[group_num] = []
            groups[group_num].append(student)

        # Convert to the original format
        result = []
        max_group = max(groups.keys())
        
        for group_num in range(0, max_group + 1):
            if group_num in groups:
                students = sorted(groups[group_num])  # Sort for consistent output
                if len(students) == 1:
                    # Single student - add as individual element
                    result.append(students[0])
                else:
                    # Multiple students - add as tuple
                    result.append(tuple(students))
        prior.append(result)
    return prior

def transform_M_EE_to_us(MyData: Data, M_in: list, print_out=False):
    M_out = np.zeros(shape=(MyData.n_stud, MyData.n_schools))

    #print(M_in)
    for j, assigned_students in enumerate(M_in):
        #print(j)
        #print(assigned_students)
        for i in assigned_students:
            M_out[i][j] = 1

    return M_out


def transform_M_us_to_EE(MyData:Data, M_in: np.ndarray, print_out = False):
    M_out = [set() for _ in range(MyData.n_schools)]

    for i in range(MyData.n_stud):
        for j in range(MyData.n_schools):
            if M_in[i][j] == 1:
                M_out[j].add(i)

    return M_out

###############################################
########### Codes Erdil & Ergin ###############
##### (with small changes to fit our code) ####
###############################################


############# GENERAL ##############

def EE(MyData: Data, step_print = True):
    # Will run Erdil & Ergin from scratch for random tie-breaking

    # Change format of preferences and priorities
    N = transform_pref_us_to_EE(MyData)
    A = transform_prior_us_to_EE(MyData)
    Q = MyData.cap
    allocation, iterations, total_moves, total_cycles, improved_students = Ergin_Erdil_Alg(N, A, Q)

    if step_print:
        print()
        print(('iterations    :', iterations))
        print(('total moves   :', total_moves))
        print(('total cycles  :', total_cycles))
        print(('lucky students:', len(improved_students)))

    return iterations, total_moves, len(improved_students), allocation    


def Ergin_Erdil_Alg(N, A, Q):
    ## Ergin-Erdil Algorithm
    A_tb = TB(A, tie_break)
    rv = DA_Erdil_ergin(N, A_tb, Q)
    rv = SIC_EE(N, A, Q, rv['stable_all'], rv['proposeoffset'])
##    if rv['iterations'] > 0 :
##        print N
##        print A_tb
##        print Q
##        printS(N)
##        printS(A)
##        printS(A_tb)
##        printS(Q)
##        printS(rv['allocation'])
##        printS(rv['proposeoffset'])
    return rv['optimal_all'], rv['iterations'], rv['total_moves'], rv['total_cycles'], rv['improved_students']

############# TIE-BREAKING ############

def nonuniform_tie_break(A):
    # Multiple Tie-Breaking
        # Random draws for each school
    new_A = []
    for school in A:
        ranks = {}
        for key,value in list(school.items()):
            ranks[key] = value + random()
        new_A.append(ranks)
    return new_A        

def uniform_tie_break(A):
    # Single Tie-Breaking
        # One random draw for each student
    n = len(A[0])
    rv = []
    for student in range(n):
        rv.append(random())
    
    new_A = []
    for school in A:
        ranks = {}
        for key,value in list(school.items()):
            ranks[key] = value + rv[key]
        new_A.append(ranks)
    return new_A

tie_break = nonuniform_tie_break

def TB(A, tb_function):
    return tb_function(A)


############# DEFERRED ACCEPTANCE ###############

def DA_Erdil_ergin(N, A, Q, print_out = False):
    (all, pro_off) = gale_shapley_Erdil_Ergin(N, A, Q, print_out)
    return {'stable_all' : all, 'proposeoffset' : pro_off}

def gale_shapley_Erdil_Ergin(N, A, Q, print_out=False):
    """
        Returns object with two elements
            'stable_all': this is list with matching 
            'proposeoffset': contains the preferences of the objects to which the agents are assigned (indexed from 0)
    """
    kickedouts = list(range(len(N)))
    proposeoffset = [0] * len(N)
    allocation = []
    for school in A:
        allocation.append(set([]))

    while(True):
        if len(kickedouts) <= 0:
            break
        if print_out:
            print() 
            print(('allocation :', allocation))
            print(('kickedouts :', kickedouts))
            print(('pro_offset :', proposeoffset))
        new_kickedouts = []
        more_to_go = False
        for student in kickedouts:
            if proposeoffset[student] >= len(N[student]):
                new_kickedouts.append(student)
                proposeoffset[student] = len(N[student])
                continue

            more_to_go = True
            school_to_apply = N[student][proposeoffset[student]]
            if print_out:
                print((student, 'applies', school_to_apply))

            rank = A[school_to_apply].get(student, None)
            if rank is None:
                proposeoffset[student] += 1
                new_kickedouts.append(student)
                continue
            
            if len(allocation[school_to_apply]) < Q[school_to_apply] :
                allocation[school_to_apply].add(student)
                if print_out:
                    print(('student', student, 'assigned to', school_to_apply))
            else :
                std = list(allocation[school_to_apply])
                students_sorted_by_rank = list(zip([A[school_to_apply][el] for el in std], std))
                students_sorted_by_rank.sort()
                students_sorted_by_rank.reverse()
                students_sorted_by_rank = [el[1] for el in students_sorted_by_rank]
                
                assigned = False
##                for other_student in allocation[school_to_apply]:
                for other_student in students_sorted_by_rank:
                    if rank < A[school_to_apply][other_student]:
                        proposeoffset[other_student] += 1
                        new_kickedouts.append(other_student)
                        allocation[school_to_apply].add(student)
                        allocation[school_to_apply].remove(other_student)
                        assigned = True
                        if print_out:
                            print(('student', student, 'assigned to', school_to_apply))
                            print(('student', other_student, 'kicked out from', school_to_apply))
                        break
                if not assigned:
                    proposeoffset[student] += 1
                    new_kickedouts.append(student)
                    if print_out:
                        print(('student', student, 'rejected from', school_to_apply))

        kickedouts = new_kickedouts
        if not more_to_go :
            break
    return (allocation, proposeoffset)


########## STABLE IMPROVEMENT CYCLES ###########

def SIC_EE(N, A, Q, allocation, pro_off, print_out = False):
    """
        Careful: always imput all variables in the format of EE (if necessary, use transformation functions)

        This function finds stable improvement cycles

        N = preferences
        A = priorities
        Q = capacities
        allocation
        'pro_off' contains the preferences of the assigned objects for the agents

    """
    all = deepcopy(allocation)
    pro = construct_proposals(N, A, pro_off, print_out) # Dictionaries with for each school, the students who prefer that school to their current school, together with their rank.

    if print_out:
        print("Pro", pro)

    improved_students = set([])
    iterations = 0
    total_moves = 0
    total_cycles = 0
    
    while(True):
        if print_out:
            print(('allocation :', all))


        ############
        # ERROR ####
        ############
        # Always computes again from the initial matching! 
        # Like this, a student might get first better off, and then again worse off, and this may cause instability.

        # Also, after executing the cycle, the proposals are wrong: some students still propose to things that are now less preferred!  

        best = best_substitudes(pro) # Retain set of students with best rank among proposers, for each school
        graph, soe = construct_digraph(all, best) # Make graph with one node for each school
        
        obj = DFS(graph)
        if print_out:
            print(('pro_off    :', pro_off)) # Preferences of assigned students
            print(('application:', pro)) # Proposing students and their ranks at each school
            print(('best-subs  :', best)) # Set of students who propose with highest rank for each school (D_j in EE paper)
            print(('graph      :', graph))
            print(('soe        :', soe))
        obj.DFS() #Find cycle
        if print_out:
            print(('Cycle', obj.cycle))
        if obj.cycle is None:
            break
        improve_allocations(N, A, all, pro, soe, obj.cycle, print_out) # Careful, these variables are changed by this function
        if print_out:
            print(('pro_off after cycle:', pro_off)) # Preferences of assigned students
            print(('application after cycle:', pro))

        iterations += 1
        total_moves += calculate_moves(N, soe, obj.cycle)
        total_cycles += len(obj.cycle)
        for index in range(len(obj.cycle)-1):
            improved_students.add(soe[(obj.cycle[index], obj.cycle[index+1])])
        improved_students.add(soe[(obj.cycle[-1], obj.cycle[0])])

    return {'optimal_all' : all, 'iterations' : iterations, \
            'total_moves' : total_moves, 'total_cycles' : total_cycles , \
            'improved_students' : improved_students}


def SIC_EE_with_mistake_EE(N, A, Q, allocation, pro_off, print_out = False):
    """
        Careful: always imput all variables in the format of EE (if necessary, use transformation functions)

        This function finds stable improvement cycles

        N = preferences
        A = priorities
        Q = capacities
        allocation
        'pro_off' contains the preferences of the assigned objects for the agents

    """
    all = deepcopy(allocation)
    pro = construct_proposals(N, A, pro_off, print_out) # Dictionaries with for each school, the students who prefer that school to their current school, together with their rank.

    if print_out:
        print("Pro", pro)

    improved_students = set([])
    iterations = 0
    total_moves = 0
    total_cycles = 0
    
    while(True):
        if print_out:
            print(('allocation :', all))


        ############
        # ERROR ####
        ############
        # Always computes again from the initial matching! 
        # Like this, a student might get first better off, and then again worse off, and this may cause instability.

        # Also, after executing the cycle, the proposals are wrong: some students still propose to things that are now less preferred!  

        best = best_substitudes(pro) # Retain set of students with best rank among proposers, for each school
        graph, soe = construct_digraph(all, best) # Make graph with one node for each school
        
        
        obj = DFS(graph)
        if print_out:
            print(('pro_off    :', pro_off)) # Preferences of assigned students
            print(('application:', pro)) # Proposing students and their ranks at each school
            print(('best-subs  :', best)) # Set of students who propose with highest rank for each school (D_j in EE paper)
            print(('graph      :', graph))
            print(('soe        :', soe))
        obj.DFS() #Find cycle
        if print_out:
            print(('Cycle', obj.cycle))
        if obj.cycle is None:
            break
        improve_allocations_with_mistake_EE(N, A, all, pro, soe, obj.cycle, print_out) # Careful, these variables are changed by this function
        if print_out:
            print(('pro_off after cycle:', pro_off)) # Preferences of assigned students
            print(('application after cycle:', pro))

        iterations += 1
        total_moves += calculate_moves(N, soe, obj.cycle)
        total_cycles += len(obj.cycle)
        for index in range(len(obj.cycle)-1):
            improved_students.add(soe[(obj.cycle[index], obj.cycle[index+1])])
        improved_students.add(soe[(obj.cycle[-1], obj.cycle[0])])

    return {'optimal_all' : all, 'iterations' : iterations, \
            'total_moves' : total_moves, 'total_cycles' : total_cycles , \
            'improved_students' : improved_students}

def construct_proposals(N, A, proposeoffset, print_out = False):
    if print_out:
        print('\nconstruct_proposals()')
    proposals = []
    for school in range(len(A)):
        proposals.append({})
        
    for student in range(len(N)):
        for school_got_rejected in N[student][:proposeoffset[student]]:
            if print_out:
                print(('\tstudent', student, 'is rejected from', school_got_rejected,N[student][:proposeoffset[student]]))
            rank = A[school_got_rejected].get(student, None)
            if rank is None:
                continue
            students_in_the_same_rank = proposals[school_got_rejected].get(rank, None)
            if students_in_the_same_rank is None:
                students_in_the_same_rank = proposals[school_got_rejected][rank] = set([])
            students_in_the_same_rank.add(student)
    return proposals

def best_substitudes(proposals):
    bests = []
    for pro_school in proposals :
        ranks = list(pro_school.keys())
        if len(ranks) <= 0:
            bests.append(set([]))
            continue
        ranks.sort()
        bests.append(pro_school[ranks[0]]) # Add only students who are ranked highest and are proposing there
    return bests

def construct_digraph(from_set, to_set): # from_set = allocation, to_set is D_j, i.e., set of highest ranked proposing students at a school
    V = list(range(len(from_set)))
    
    graph = []
    for vertex in V:
        graph.append(set([]))
    students_on_the_edge = {}

    for v in V:
        fs = from_set[v] # fs is assigned students in school v
        for student in fs:
            for w in V :
                ts = to_set[w] # ts are students who are in highest ranked group among proposers at school w
                if student in ts:
                    graph[v].add(w) # edge between school v and w is there is a student assigned to v who is in D_w
                    students_on_the_edge[(v,w)] = student # with student 'student' preferring to be assigned to v instead of w
                    
    return graph, students_on_the_edge

class DFS:
    def __init__(self, G):
        self.G = G
        self.V = list(range(len(G)))
        self.cycle = None
        self.reset()

    def reset(self):
        # 0: white, 1:gray, 2:black
        self.color = [0] * len(self.V)
        self.pi = [None] * len(self.V)

    def DFS(self, print_out = False):
        if print_out:
            print('DFS()')
        for u in self.V:
            if (self.color[u] == 0) and (self.cycle is None):
                self.DFSVisit(u)

    def DFSVisit(self, u, print_out = False):
        self.color[u] = 1
        for v in self.G[u]:
            if self.cycle is not None:
                return
            if self.color[v] == 0:
                self.pi[v] = u
                self.DFSVisit(v)
            elif self.color[v] == 1:
                # there is a cycle
                path = [u]
                while (True):
##                    print path
                    if self.pi[path[-1]] is None:
                        raise RuntimeError(f"Parent is None while building cycle at node {path[-1]} — likely missing assignment in DFS")

                    path.append(self.pi[path[-1]])
                    if path[-1] == v:
                        path.reverse()
                        if print_out:
                            print(('found a cycle', path))
                        self.cycle = path
                        break
                    
        self.color[u] = 2


def improve_allocations(N, A, allocation, proposals, soe, cycle, print_out = False):
    if cycle is None or len(cycle) < 2 :
        return
    def move(school1, school2):
        student = soe[(school1, school2)]
        if print_out:
            print(('\tmove', school1, school2, 'student:', student))
        allocation[school1].remove(student)
        allocation[school2].add(student)
        rank = A[school2][student]
        #proposals[school2][rank].remove(student) # Removes student from their newly assigned school in proposals

        if len(proposals[school2][rank]) == 0:
            proposals[school2].pop(rank)

        ####### MODIFICATION IN CODE ERDIL & ERGIN ######
        # IMPORTANT: Also remove the student from all schools they wanted that are more preferred than their old school,
        # but less preferred than their new school

        pref_index_old_school = N[student].index(school1)
        pref_index_new_school = N[student].index(school2)

        if print_out:
            print('pref_index_old_school', pref_index_old_school)
            print('pref_index_new_school', pref_index_new_school)

        for i in range(pref_index_new_school, pref_index_old_school): # '+1' because we already removed the student from the new school

            school3 = N[student][i]
            rank = A[school3][student]
            if print_out:
                print("Rank student", student, "at school", school3, ":", rank)
                print("Proposals school", school3, ":", proposals[school3])
                print("Proposals school", school3, ":", proposals[school3][rank])
                print("student", student)
                print("Proposals", proposals)
                
            proposals[school3][rank].remove(student)

            if len(proposals[school3][rank]) == 0:
                proposals[school3].pop(rank)

        # END MODIFICATIONS
        ###################

        
        
    for index in range(len(cycle)-1):
        move(cycle[index], cycle[index+1])
    move(cycle[-1], cycle[0])


def improve_allocations_with_mistake_EE(N, A, allocation, proposals, soe, cycle, print_out = False):
    if cycle is None or len(cycle) < 2 :
        return
    def move(school1, school2):
        student = soe[(school1, school2)]
        if print_out:
            print(('\tmove', school1, school2, 'student:', student))
        allocation[school1].remove(student)
        allocation[school2].add(student)
        rank = A[school2][student]
        proposals[school2][rank].remove(student) # Removes student from their newly assigned school in proposals

        if len(proposals[school2][rank]) == 0:
            proposals[school2].pop(rank)        
        
    for index in range(len(cycle)-1):
        move(cycle[index], cycle[index+1])
    move(cycle[-1], cycle[0])

def calculate_moves(N, soe, cycle, print_out = False):
    if cycle is None or len(cycle) < 2 :
        return 0
    def move(school1, school2):
        student = soe[(school1, school2)]
        rv = N[student].index(school1) - N[student].index(school2)
        if print_out :
            print(('\tstudent', student, 'improved', rv, 'in his/her preference list'))
        return rv

    if print_out:
        print(('improvement cycle', cycle, 'details:'))
    moves = 0
    for index in range(len(cycle)-1):
        moves += move(cycle[index], cycle[index+1])
    moves += move(cycle[-1], cycle[0])
    return moves
