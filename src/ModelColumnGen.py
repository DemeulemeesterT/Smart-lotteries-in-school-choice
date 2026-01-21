from .Assignment import *
from .Assignment import find_identical_students_from_matrix
from .Assignment import stability_test_single_matching
from .SICs import *
from .DA_STB import *

# Install necessary packages before running the script:
# pip install pulp
# pip install gurobipy
# pip install pyscipopt
# Note: Obtain an academic license for Gurobi from: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

# Check with solvers available on computer
import pulp as pl
from pulp import *

import time

import gurobipy

from contextlib import redirect_stdout

# To check which solvers available on computer:
# print(pl.listSolvers(onlyAvailable=True))

class SolutionReport:
    """
    Will be output by Solve function. Will be saved using pickle, and analyzed afterwards
    """
    # Information added to solution report during data generation
    Data: Data          # Contains data instance
    n_stud: int
    n_schools: int
    alpha: float        # alpha and beta will only be filled in when using data generation Erdil & Ergin
    beta: float
    seed: int

    # Information added after solution is found
    A: Assignment       # Contains the final assignment
    A_SIC: Assignment   # Contains warm start solution (in general, SICs by Erdil & Ergin)
    A_DA_prob: np.ndarray # Assignment probabilities to sd-dominate (in general, DA)
    avg_ranks: dict     # Contains average ranks of several solutions along the process
    n_students_assigned: float # Number of students assigned in the final solution
    obj_master: list    # Objective values of master in iterations
    obj_pricing: list   # Objective values of pricing in iterations
    n_iter: int         # Number of iterations
    n_match: int        # Number of matchings
    n_match_support: int # Number of matchings in support (positive weight) of final solution
    time_limit_exceeded: bool # Whether time limit is exceeded
    time_limit: int     # Time limit
    optimal: bool       # Optimality guaranteed?
    time: float         # Time used
    #Xdecomp: list       # Matchings in the found decomposition
    #Xdecomp_coeff: list # Weights of these matchings
    bool_ColumnGen: bool # True if column generation if performed, False if only first step is performed
    #... 





class ModelColumnGen: 
    """
    Contains two methods:
        __init__: initializes the model, and the solver environment

        Solve: solves the model.
            The parameters of this method can control which objective function is optimized, and which solver is used
    """
    
    # Used this example as a template for Pulp: https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
    
    def __init__(self, MyData: Data, p: Assignment, p_DA: np.ndarray, bool_identical_students: bool, bool_punish_unassigned:bool, print_out: bool):
        """
        Initialize an instance of Model.

        Args:
            MyData (type: Data): instance of class Data.
            p (type: Assignment): instance of class Assignment (possibly including SICs). This will be used for warm start solution
            p_DA (type: np.ndarray): this is the probabilistic assignment which we want to sd_dominate
            bool_identical_students (bool): if True, give identical students the same probabilities
            bool_punish_unassigned (bool): if True, give utility of 'n_schools' to being unassigned
            print_out (type: bool): boolean that controls which output is printed.
        """
        self.MyData = copy.deepcopy(MyData)
        #self.p = copy.deepcopy(p)
        self.p = p
        #self.p_DA = copy.deepcopy(p_DA)
        self.p_DA = p_DA.copy()
        self.bool_identical_students = bool_identical_students
        self.bool_punish_unassigned = bool_punish_unassigned


        # Create the pulp model
        # 'self.master' refers to master problem
        # 'self.pricing' will refer to pricing problem
        self.master = LpProblem("Improving_ex_post_stable_matchings", LpMinimize)

        # Create variables to store the solution in
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings
        zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
        self.Xassignment = Assignment(MyData, zero) # Contains the final assignment found by the model

        # Ranges that will help with coding
        self.STUD = range(0,self.MyData.n_stud)
        self.SCHOOLS = range(0, self.MyData.n_schools)

        # Tuple with all student-school pairs that are preferred to outside option
        # This tuple contains the INDICES of the students and the pairs, and not their original names
        self.PAIRS = []
        for i in range(0, MyData.n_stud):
            for j in range(0,len(MyData.pref[i])):
                # Convert pref[i][k] (school ID as string) to column index
                col_index = MyData.ID_school.index(MyData.pref[i][j])
                self.PAIRS.append((i,col_index))   
        
        # First, convert the set containing all matchings to a list to make it subscribable
        self.M_list = list(p.M_set)

        # M[k][i][j] = 1 if student i is assigned to school j in matching k, and 0 otherwise
        self.nr_matchings = len(self.M_list)
        self.N_MATCH = range(self.nr_matchings)  # Number of matchings

        # Create the parameter set M[k][i][j]
        self.M = np.zeros((self.nr_matchings, self.MyData.n_stud, self.MyData.n_schools))
        for k in self.N_MATCH:
            for i in range(self.MyData.n_stud):
                for j in range(self.MyData.n_schools):
                    self.M[k, i, j] =self.M_list[k][i][j]  # Fill the parameter from the M_list

        #self.M = LpVariable.dicts("M", [(k, i, j) for k in self.N_MATCH for i, j in self.PAIRS], cat="Binary")

        # Store labels to make understanding output easier
        #self.labels = {}
        #for k in self.N_MATCH:
        #    for i in range(self.MyData.n_stud):
        #        for j in range(self.MyData.n_schools):
        #            student_name = self.MyData.ID_stud[i]
        #            school_name = self.MyData.ID_school[j]
        #            self.labels[k, i, j] = f"M_{k}_{student_name}_{school_name}"



        #### OBJECTIVE FUNCTION ####
        # Add an empty objective function
            # Every time you update it, you should add it to the model again
                # using a code like:
                # self.model.setObjective(self.model.objective+obj_coeff*self.w[m])
        self.master += LpAffineExpression()
            
            
            
            
        #### CONSTRAINTS ####
        # Other constraints defined for specific models in functions below (see function Solve)
        
        self.constraints = {}

        # Ensure weights sum up to one
        # We save this constraint in order to later add decision variables to it.
        self.constraints["Sum_to_one"] = LpConstraintVar("Sum_to_one", LpConstraintEQ, 1)

        # Add one constraint for each pair to model first-order stochastic dominance
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                school_name = self.MyData.pref_index[i][j]
                # Compute original probability of being assigned to j-th ranked school or better
                original_p = 0
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    original_p += self.p_DA[i,pref_school]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                self.constraints[name]=LpConstraintVar(name, LpConstraintGE, original_p)

        # Non-negativity (explicitly included to get dual variables)
        #for m in self.N_MATCH:
        #    name = 'GE0_' + str(m)
        #    self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
        
        # Add constraints for identical students, if required
        if self.bool_identical_students:
            self.identical_students = find_identical_students_from_matrix(self.p_DA, self.MyData, print_out)
            for [i,j] in self.identical_students:
                sum = 0
                for p in range(len(self.MyData.pref[i])):
                    school_index = self.MyData.pref_index[i][p]
                    sum += sum + self.p_DA[i][school_index]
                    name = "IDENTICAL_" +  str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + 'a'
                    self.constraints[name] = LpConstraintVar(name, LpConstraintLE, 0.05)
                    name = "IDENTICAL_" +  str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + 'b'
                    self.constraints[name] = LpConstraintVar(name, LpConstraintLE, 0.05)
                    if sum >= 1 - 0.0000001:
                        break

        # Add all these contraints to the model
        for c in self.constraints.values():
            self.master += c

        

        #### DECISION VARIABLES ####
        self.w = []
        #print(self.nr_matchings)
        for m in tqdm(self.N_MATCH, desc='Master: add decision variables', unit='var', disable= not print_out):
            #print(self.M[m])
            self.add_matching(self.M[m], m, print_out)



        # Create variables, and add them to the constraints
        
        # w[k] is the weight of matching k in the decomposition
        #self.w = LpVariable.dicts("w", self.N_MATCH, 0, 1,  )
        #self.w = LpVariable("w", self.nr_matchings, lowBound=0, upBound=1, e={self.constraints["Sum_to_one"]:1} )

        #self.w = LpVariable("w", self.nr_matchings, e={self.constraints["Sum_to_one"]:1} )
        #self.master += 2*self.w

        
        #self.vars += self.w()
        #for m in self.N_MATCH:
        #    for c in range(1,len(self.constraints)):
        #        self.master.addVariableToConstraints(self.w[m], {self.constraints[c], 1})

        # self.master.writeLP("TestColumnFormulation.lp")

        
        # Set the warm start solution as the decomposition found after SICs
        for m in self.N_MATCH:
            # Find matching (because w_set is a set and not subscriptable)
            M = self.M_list[m]
            self.w[m].setInitialValue(self.p.w_set[M])
        
        #self.master.writeLP("TestColumnFormulation.lp")
        
        
      
    def add_matching(self, M_in: np.ndarray, index, print_out: bool):
        """
        Function to add a matching M as a decision variable to the master problem
        Index is the index of the matching in the master problem
        """  
        
        # First, determine coefficients of this variable in the constraints
        coeff = {}
        coeff["Sum_to_one"] = 1
        coeff["Obj"] = 0 # Objective coefficients will be fixed later
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                school_name = self.MyData.pref_index[i][j]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                
                # Initialize the coefficient if it doesn't exist
                # (needed because we use +=, and not =)
                if name not in coeff:
                    coeff[name] = 0
                
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    #print(m,i,j,pref_school, self.M[m,i,pref_school])
                    coeff[name] += M_in[i,pref_school]
        
        if self.bool_identical_students:
            for [i,j] in self.identical_students:
                sum = 0
                for p in range(len(self.MyData.pref[i])):
                    school_index = self.MyData.pref_index[i][p]
                    sum += sum + self.p_DA[i][school_index]
                    name = "IDENTICAL_" +  str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + "a"

                    # Initialize the coefficient if it doesn't exist
                    # (needed because we use +=, and not =)
                    if name not in coeff:
                        coeff[name] = 0

                    # Note that if a pair [i,j] belongs to identical students, then i < j.
                    # That's why we put the minus sign for j, to enfore that they should be equal
                    coeff[name] += M_in[i, school_index]
                    coeff[name] -= M_in[j, school_index]

                    name = "IDENTICAL_" +  str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + "b"

                    # Initialize the coefficient if it doesn't exist
                    # (needed because we use +=, and not =)
                    if name not in coeff:
                        coeff[name] = 0

                    # Note that if a pair [i,j] belongs to identical students, then i < j.
                    # That's why we put the minus sign for j, to enfore that they should be equal
                    coeff[name] -= M_in[i, school_index]
                    coeff[name] += M_in[j, school_index]

                    if sum >= 1 - 0.0000001:
                        break

        # Non-negativity of the variables
        # First, add a new constraint, and only then set the coefficient
        #for m in self.N_MATCH: 
        #    name = 'GE0_' + str(m)
        #    coeff[name] = 0
        #name = 'GE0_' + str(index)
        #coeff[name] = 1
        
        # Then, create a dictionary for `e` that maps constraints to their coefficients
        #e_dict = {self.constraints[key]: coeff[key] for key in self.constraints if coeff[key] > 0}
        e_dict = {self.constraints[key]: coeff[key] for key in self.constraints if coeff[key] != 0}

        # Add this variable to self.w
        name_w = "w_" + str(index)
        self.w.append(LpVariable(name_w, lowBound= 0, e=e_dict))
        
                        
        # Compute objective coefficient of this variable (average rank)
        if not self.bool_punish_unassigned:
            obj_coeff = 0
            for (i,j) in self.PAIRS:
                obj_coeff += M_in[i,j]*(self.MyData.rank_pref[i,j]+ 1) / self.MyData.n_stud # + 1 because the indexing starts from zero

            
        else:
            # If we punish being unassigned, give utility of 'n_schools' to being unassigned
            obj_coeff = self.MyData.n_schools
            for (i,j) in self.PAIRS:
                obj_coeff -= M_in[i,j]*(self.MyData.n_schools - (self.MyData.rank_pref[i,j]+ 1)) / self.MyData.n_stud # + 1 because the indexing starts from zero
                #if print_out:
                #    print("Objective coefficient", obj_coeff)

        # Add this variable to the model with the correct objective coefficient
        self.master.setObjective(self.master.objective+obj_coeff*self.w[index])
        
        #self.N_MATCH = range(len(self.M_list))
        #print(self.N_MATCH)

        #self.master.writeLP("TestColumnFormulation.lp")




        
    def Solve(self, stab_constr: str, solver: str, print_log: str, time_limit: int, n_sol_pricing: int, n_sol_pricingMinRank: int, gap_solutionpool_pricing: float, MIPGap: float, bool_ColumnGen: bool, bool_supercolumn: bool, print_out: bool):
        """
        Solves the formulation using column generation.
        Returns an instance from the Assignment class.

        Note that, if you create an object of ModelColumnGen using an assignment object containing SICs already,
        then those will be the matchings that are used.

        Args:
            stab_constr (str): controls which type of stability constraints are used.
            solver (str): controls which solver is used. See options through following commands:
                solver_list = pl.listSolvers(onlyAvailable=True)
                print(solver_list)
            print_log: print output of solver on screen?
            time_limit: in s
            n_sol_pricing: number of solutions returned by pricing problem
            n_sol_pricingMinRank: number of solutions returned by the pricing problem that minimizes the average rank
            gap_solutionpool_pricing: optimality gap used for the solutions included in the solution pool in the pricing problem
            MIPGap: gap for pricing problem
            print_out (bool): boolean that controls which output is printed.
            bool_ColumnGen (bool): if True: perform entire column generation for time_limit period
                            if False: only perform first iteration, and don't build pricing problem
            bool_supercolumn (bool): if True: if the model is infeasible in the first step, add an artificial matching
                M' such that M'[i][j] = 1 for all i,j. Give this matching a very high weight in the objective function,
                and remove it as soon as a feasible solution can be found where the weight of this 'matching' is zero.
        """
        self.bool_ColumnGen = bool_ColumnGen
        self.stab_constr = stab_constr
        
        # Compute average rank of current assignment
        self.avg_rank_DA = 0
        for (i,j) in self.PAIRS:
            self.avg_rank_DA += self.p_DA[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
        # Average
        self.avg_rank_DA = self.avg_rank_DA/self.MyData.n_stud
        #if print_out == True:  

        self.avg_rank = 0
        for (i,j) in self.PAIRS:
            self.avg_rank += self.p.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
        # Average
        self.avg_rank = self.avg_rank/self.MyData.n_stud
        if print_out:   
            print(f"\nAverage rank DA : {self.avg_rank_DA}.\n")
            print(f"\nAverage rank warm start solution : {self.avg_rank}.\n\n")
        
        
        # Check that strings-arguments are valid

        # Valid values for 'solver'
        solver_list = pl.listSolvers(onlyAvailable=True)
        if solver not in solver_list:
           raise ValueError(f"Invalid value: '{solver}'. Allowed values are: {solver_list}")

        # Valid values for 'stab_constr'
        stab_list = ["TRAD", "CUTOFF"]
        if stab_constr not in stab_list:
           raise ValueError(f"Invalid value: '{stab_constr}'. Allowed values are: {stab_list}")        

        self.time_limit_exceeded = False
        self.time_limit = time_limit

        #### PRICING PROBLEM ####
        # Defined in separate function
        if bool_ColumnGen == True:
            self.build_pricing(stab_constr, print_out)

            #self.pricing.writeLP("PricingProblem.lp")


        
        #### RUN COLUMN GENERATION PROCEDURE ####
        optimal = False
        
        #self.pricing.writeLP("PricingProblem.lp")
        self.iterations = 1
        if print_out:
            print("Number of matchings:", self.nr_matchings)

        # Create two empty arrays to store objective values of master and pricing problem
        self.obj_master = []
        self.obj_pricing = []

        starting_time = time.monotonic()

        self.supercolumn_in_model = False
        self.index_super_column = -1 # Updated in loop below if supercolumn added
        remove_supercolumn = False # Used to decide when to remove

        # Keep track of whether the supercolumn is in the master problem
        # If it is, check whether its weight in the found solution is zero, in which case the variable will be removed

        while (optimal == False):
            if print_out:
                print('\nITERATION:', self.iterations)            
            if print_out:
                if print_out:
                    print("\n ****** MASTER ****** \n")
                #for m in self.N_MATCH:
                #    print(self.M_list[m])

            # String can't be used as the argument in solve method, so convert it like this:
            solver_function = globals()[solver]  # Retrieves the GUROBI function or class
        
            #self.master.writeLP("TestColumnFormulation.lp")

            # Solve the formulation
            if print_log == False:
                self.master.solve(solver_function(msg = False, logPath = "Logfile_master.log", warmStart = True))
            else:
                self.master.solve(solver_function(msg = True, warmStart = True))
            #self.model.solve(GUROBI_CMD(keepFiles=True, msg=True, options=[("IISFind", 1)]))
            
            # Get objective value master problem
            self.obj_master.append(self.master.objective.value())
            if print_out:
                print("Objective master: ", self.obj_master[-1])

            # Get average rank of first iteration
            if self.iterations == 1:
                self.avg_rank_first_iter = self.obj_master[-1]

            # Check if the time limit is exceeded
            current_time = time.monotonic()

            #if print_out:
            #    for m in self.N_MATCH:
            #        print("w_", m, self.w[m].value())
            
        

            if bool_ColumnGen == False: # You don't want to run entire column generation procedure
                # Create solution report
                self.time_limit_exceeded = False
                optimal = False
                self.time_columnGen = current_time - starting_time
                return self.generate_solution_report(print_out) 

            elif bool_ColumnGen == True: # Only if you want to continue with the column generation procedure.
                # If infeasible, add "supercolumn"
                # This is a "matching" M such that M[i][j] = 1 for all i,j
                # This will help to find a solution that sd-dominates the given random matching
                # To avoid selecting it, make objective value of it very large.
                # For instance, if we do not want to select is with probability 0.0001, 
                    # give it a weight of n^2*10000, for example
                if self.iterations == 1:
                    if self.master.status == -1: # If master is infeasible
                        if bool_supercolumn == True:
                            if print_out:
                                print("\n Supercolumn added to model to enforce feasibility.\n")
                            M_super = np.ones(shape=(self.MyData.n_stud, self.MyData.n_schools))
                            self.index_super_column = len(self.w)

                            
                            self.M_list.append(M_super)
                            self.nr_matchings =self.nr_matchings + 1
                            self.N_MATCH = range(self.nr_matchings)
                            self.add_matching(M_super, len(self.w), False)
   
                            self.supercolumn_in_model = True

                            # Modify objective coefficient:
                            obj_coeff = self.MyData.n_stud * self.MyData.n_stud * 10000
                            self.master.setObjective(self.master.objective+obj_coeff*self.w[self.index_super_column])

                            #self.master.writeLP("TestColumnFormulation.lp")

                            # Now solve the model again
                            if print_log == False:
                                self.master.solve(solver_function(msg = False, logPath = "Logfile_master.log", warmStart = True))
                            else:
                                self.master.solve(solver_function(msg = True, warmStart = True))
                            #self.model.solve(GUROBI_CMD(keepFiles=True, msg=True, options=[("IISFind", 1)]))
                            
                            # Get objective value master problem
                            self.obj_master.append(self.master.objective.value())
                            if print_out:
                                print("Objective master: ", self.obj_master[-1])

                            # Get average rank of first iteration
                            if self.iterations == 1:
                                self.avg_rank_first_iter = self.obj_master[-1]


                # Check whether supercolumn has weight zero in solution, and remove it in that case
                # You can only do this after pricing problem, because otherwise the model has been changed
                    # and dual variables cannot be extracted.

                if self.master.status == -1: # If master is infeasible
                    # Create solution report
                    self.time_limit_exceeded = False
                    optimal = False
                    self.time_columnGen = current_time - starting_time
                    return self.generate_solution_report(print_out) 
                
                if current_time - starting_time > time_limit:
                    optimal = True
                    self.time_limit_exceeded = True
                    self.time_columnGen = current_time - starting_time
                    return self.generate_solution_report(print_out)
                
                ## SOLVE PRICING ####
                # Get dual variables
                duals = {}

                self.max_dual = 0
                
                duals["Sum_to_one"] = self.master.constraints["Sum_to_one"].pi

                for i in self.STUD:
                    for j in range(len(self.MyData.pref[i])):
                        school_name = self.MyData.pref_index[i][j]
                        name_duals = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                        name_constr = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                            
                        duals[name_duals]=self.master.constraints[name_constr].pi

                        if duals[name_duals] > self.max_dual:
                            self.max_dual = duals[name_duals]

                        if abs(duals[name_duals] < 0.00001): # Get rid of small numerical inaccuracies
                            duals[name_duals] = 0

                        #if duals[name_duals] > 0.00001:
                        #if print_out:
                        #    print(name_duals, duals[name_duals])
                                        #for m in self.N_MATCH:
                if self.max_dual == 0:
                    self.max_dual = 0.001 # Later, we divide by this value
                #    name_GE = 'GE0_' + str(m)
                #    duals[name_GE] = self.master.constraints[name_GE].pi
                #if print_out:
                #    print(duals)
                
                #for name in duals:
                    #if duals[name] > 0:
                        #print(name, duals[name])
                    
                # Modify objective function pricing problem
                pricing_obj = LpAffineExpression()
                for i in self.STUD:
                    #print('student ', i)
                    for j in range(len(self.MyData.pref[i])): 
                        school_name = self.MyData.pref_index[i][j]
                        if self.bool_punish_unassigned == False:
                            pricing_obj += self.M_pricing[i,school_name] * ( - (self.MyData.rank_pref[i,school_name]+ 1) / self.MyData.n_stud ) # + 1 because the indexing starts from zero
                        else:   
                            raise ValueError(f"The pricing problem with punishing unassigned students is not yet implemented. Please set bool_punish_unassigned to False.")
                        
                        for k in range(j+1):
                            pref_school = self.MyData.pref_index[i][k]
                            name = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(pref_school)
                            pricing_obj += self.M_pricing[i,pref_school] * duals[name] 


                if self.bool_identical_students:
                    for [i,j] in self.identical_students:
                        sum = 0
                        for p in range(len(self.MyData.pref[i])):
                            school_index = self.MyData.pref_index[i][p]
                            sum += sum + self.p_DA[i][school_index]
                            name_duals = "Nu_" + str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + 'a'
                            name_constr = "IDENTICAL_" +  str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + 'a'
                            duals[name_duals] = self.master.constraints[name_constr].pi
                            pricing_obj += self.M_pricing[i, school_index] * duals[name_duals]
                            pricing_obj -= self.M_pricing[j, school_index] * duals[name_duals]

                            name_duals = "Nu_" + str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + 'b'
                            name_constr = "IDENTICAL_" +  str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_stud[j]) + "_" + str(self.MyData.ID_school[school_index]) + 'b'
                            duals[name_duals] = self.master.constraints[name_constr].pi
                            pricing_obj -= self.M_pricing[i, school_index] * duals[name_duals]
                            pricing_obj += self.M_pricing[j, school_index] * duals[name_duals]

                            if sum >= 1 - 0.0000001:
                                break


                #if print_out:
                #    print(pricing_obj)
                #print("Duals Sum_to_one", duals["Sum_to_one"])
                #print("Duals", duals)

                # Add the constant terms!
                constant = 0
                constant += duals['Sum_to_one']
                pricing_obj += constant
                    # This constant term will not be printed in the objective function in the .lp file
                    # As Gurobi seems to rely on the lp file, constant term doesn't appear in outputs Gurobi
                    # However, they will be included in the returned objective value by Pulp!

                
                # Use other pricing objective function
                #self.pricing.objective = LpAffineExpression()
                #self.pricing.setObjective(pricing_obj)
                
                # Pricing problem that minimizes average rank, while taking dual variables into account

                # Add constraint to enforce that reduced cost is negative (or objective pricing positive)
                self.pricing += (pricing_obj >= 0, "PricingObjConstr") 

                self.pricing.sense = pl.LpMinimize
                # Change objective function to minimize rank
                new_obj = LpAffineExpression()
                for (i,j) in self.PAIRS:
                    name_constr = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(j)
                    dual_value=self.master.constraints[name_constr].pi
                
                    new_obj += self.M_pricing[i,j]*(self.MyData.rank_pref[i,j]+ 1 +self.obj_master[-1] * dual_value/self.max_dual) / self.MyData.n_stud # + 1 because the indexing starts from zero
                    #new_obj += self.M_pricing[i,j]*(self.MyData.rank_pref[i,j] + 1) / self.MyData.n_stud # + 1 because the indexing starts from zero
            
                
                # Add this variable to the model with the correct objective coefficient
                self.pricing.objective = LpAffineExpression() 
                self.pricing.setObjective(new_obj)

                # Intermezzo: now that we have the dual variables, we can check whether the supercolumn can be removed
                    # Remove if it had a weight of zero in master problem
                #print("\n\n Supercolumn check! In model?", supercolumn_in_model)
                if self.supercolumn_in_model == True:
                    #print("index supercolumn: ", index_super_column)
                    #print("value supercolumn: ", self.w[index_super_column].varValue)
                    if self.w[self.index_super_column].varValue < 0.0001:
                        if print_out:
                            print("\n The supercolumn will be removed from the master problem.\n")
                            # This can only be done when matchings are added to the model
                            # If we would do it now, then we cannot extract weights of matchings if solution is optimal
                        remove_supercolumn = True
                        
                
                #self.pricing.writeLP("PricingProblem.lp")

                #for m in self.N_MATCH:
                #    name_GE = 'GE0_' + str(m)
                    #constant += duals[name_GE]
                if print_out:
                    print("Constant term", constant)
                
                # Solve modified pricing problem
                if print_out:
                    print("\n ****** PRICING ****** \n")
                
                constant_str = str(constant)
                
                # Update time limit:
                current_time = time.monotonic()
                new_time_limit = max(time_limit - (current_time - starting_time), 0)

                if new_time_limit < 0:
                    return self.generate_solution_report(print_out) 
                
                if print_out:
                    print('New time limit', new_time_limit)

                if print_log == True:  
                    #self.pricing.solve(solver_function())
                    
                    #print("\n\n Careful, pricing stops at first postive value!\n\n")
                    #self.pricing.solve(solver_function(timeLimit = new_time_limit, BestObjStop = -constant +0.0001))
                    self.pricing.solve(solver_function(timeLimit = new_time_limit,
                    PoolGap = gap_solutionpool_pricing,
                    PoolSolutions = n_sol_pricing,
                    #BestObjStop = -constant +0.0001,
                    PoolSearchMode = 2, #Find diverse solutions
                    MIPGap = MIPGap)) 

                    # Will stop the solver once a matching with objective function at least zero has been found
                    #self.pricing.solve(solver_function())

                else:
                    #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
                    with open(os.devnull, 'w') as devnull:
                        with redirect_stdout(devnull):
                            #self.pricing.solve(solver_function(msg=False, timeLimit=new_time_limit, logPath = 'Logfile_pricing.log',BestObjStop = -constant + 0.0001))
                            #print("\n\n Careful, pricing stops at first postive value!\n\n")
                            self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log',timeLimit = new_time_limit,
                                #BestObjStop = -constant +0.0001,
                                PoolGap = gap_solutionpool_pricing,
                                PoolSolutions = n_sol_pricing,
                                PoolSearchMode = 2,
                                MIPGap = MIPGap))
                    #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
                

                # If not infeasible:
                
                if self.pricing.status not in [0,-1]: # -1 is infeasible, 0 is unsolved (for example because interrupted earlier, or timelimit reached)
                    #print("Status code: ", self.pricing.status)
                    obj_pricing_var = self.pricing.objective.value()
                    self.obj_pricing.append(obj_pricing_var)
                    if print_out:
                        print("self.obj_pricing_var: ", obj_pricing_var)
                        print("Constant term:", constant)
                        print("\t\tObjective pricing: ", obj_pricing_var)

                #if print_out:
                if False:
                    for (i,j) in self.PAIRS:
                        print("M[",i,j,'] =', self.M_pricing[i,j].value())
                
                #### EVALUATE SOLUTION ####
                if print_out:
                    print('Pricing status', self.pricing.status)
                if self.pricing.status == 0: # Time limit exceeded
                    self.time_limit_exceeded = True
                    optimal = False
                    self.time_columnGen = self.time_limit
                    return self.generate_solution_report(print_out) 
                
                elif self.pricing.status != -1:   
                    if obj_pricing_var > 0:
                        # Remove supercolumn if needed:
                        if self.supercolumn_in_model == True:
                            if remove_supercolumn == True:
                                self.w[self.index_super_column].upBound = 0
                                #self.master.writeLP("TestColumnFormulation2.lp")

                                self.supercolumn_in_model = False


                        # The solution of the master problem is not optimal over all weakly stable matchings
                        
                        # Add non-negativity constraint to the master for this new matching
                        #name = 'GE0_' + str(len(self.w))
                        #self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
                        #self.master += self.constraints[name]

                        # Go through all generated solutions.
                        # To do this, first we should extract the gurobi model and access the solution pool through there.
                        pricing_gurobi = self.pricing.solverModel

                        n_sol_found = pricing_gurobi.SolCount

                        if print_out:
                            print('Solutions found by pricing:', n_sol_found)
                            #if stab_constr == "CUTOFF":
                            #    for i in self.SCHOOLS:
                            #        print("Cutoff school", i, ":", self.t[i].varValue)
                        
                        # Binary vector to remember which matchings were improved
                        counter_M_improved = np.zeros(n_sol_found)

                        # Go through all matchings
                        for t in tqdm(range(n_sol_found), desc='Found matchings added to master', unit='match', disable= not print_out):

                            pricing_gurobi.setParam('SolutionNumber', t)

                            # Add the matching found by the pricing problem to the master problem       
                            found_M = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

                            # Find all variables by name
                            for (i,j) in self.PAIRS:
                                student_name = self.MyData.ID_stud[i]
                                school_name = self.MyData.ID_school[j]
                                name_var = f"M_{student_name}_{school_name}"
                                gurobi_var = pricing_gurobi.getVarByName(name_var)
                                found_M[i][j] = gurobi_var.Xn

                            self.M_list.append(found_M)
                            self.nr_matchings =self.nr_matchings + 1
                            self.N_MATCH = range(self.nr_matchings)
                            self.add_matching(found_M, len(self.w), print_out)
                            #if print_out:
                            #    print(found_M)
                            #self.master.writeLP("TestColumnFormulation.lp")
                        
                            # Test stability of matching
                            #is_stable = stability_test_single_matching(self.MyData, found_M, print_out)
                            #if not is_stable:
                            #    # If not stable, print values
                            #    if self.stab_constr == "CUTOFF":
                            #        if print_out:
                            #            print("Matching found by pricing is not stable! Cutoffs:")
                            #        for i in self.SCHOOLS:
                            #            if print_out:
                            #                print("Cutoff school", i, ":", self.t[i].varValue)
                            #        for i in self.SCHOOLS:
                            #            if print_out:
                            #                print("Capacity filled school", i, ":", self.f[i].varValue)

                            
                            # Exclude this matching from being find by the pricing problem in the future.
                            self.pricing += lpSum([self.M_pricing[i,j] * found_M[i][j] for (i,j) in self.PAIRS]) <= lpSum([found_M[i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{self.nr_matchings-1}"
                    
                            # Find SICs
                            M_SIC = SIC(self.MyData, found_M, False)
                            if not np.array_equal(M_SIC, found_M): # If improvement realized by executing SICs
                                counter_M_improved[t] += 1

                                self.M_list.append(M_SIC)
                                self.nr_matchings += 1
                                self.N_MATCH = range(self.nr_matchings)
                                self.add_matching(M_SIC, len(self.w), print_out)
                                #self.master.writeLP("TestColumnFormulation.lp")
                                
                                # Exclude this matching from being find by the pricing problem in the future.
                                self.pricing += lpSum([self.M_pricing[i,j] * M_SIC[i][j] for (i,j) in self.PAIRS]) <= lpSum([M_SIC[i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{self.nr_matchings-1}"
                            #
                            #    if print_out:
                            #        # Compute reduced cost of this new matching:
                            #        M_obj = 0
                            #        for i in self.STUD:
                            #            #print('student ', i)
                            #            for j in range(len(self.MyData.pref[i])): 
                            #                school_name = self.MyData.pref_index[i][j]
                            #                pricing_obj -= self.M_pricing[i,school_name] * (self.MyData.rank_pref[i,school_name]+ 1) / self.MyData.n_stud # + 1 because the indexing starts from zero
                            #                #print('  school ', school_name, -(self.MyData.rank_pref[i,school_name]+ 1) / self.MyData.n_stud)
                            #                for k in range(j+1):
                            #                    pref_school = self.MyData.pref_index[i][k]
                            #                    name = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(pref_school)
                            #                    M_obj += M_SIC[i][pref_school] * duals[name]
                            #                    #print('      school ', pref_school, duals[name])
                            #        M_obj += duals['Sum_to_one']
                            #        print("\tObjective function new matching: ", M_obj, " (was ", pricing_gurobi.PoolObjVal + constant, ").")

                            if print_out:  
                                if t == 0:
                                    # Weirdly, the poolobjval does not take into account the constant value, 
                                    # while computing the objective value using pricing.objective.value() does?        
                                    #print("Constant value:", constant)
                                    #print("Pricing objective:", pricing_gurobi.PoolObjVal)
                                    #print("Computed objective value matching ", t, ": ", pricing_gurobi.PoolObjVal + constant)               
                                    print("Matching ", self.nr_matchings, "Objective value pricing:", pricing_gurobi.PoolObjVal + constant)

                                    
                                elif t == n_sol_found - 1:
                                    print("Matching ", self.nr_matchings, "Objective value pricing:", pricing_gurobi.PoolObjVal + constant)

                        if print_out:
                            print("New number of matchings:", self.nr_matchings)
                            #print("Improving matchings found by SICs", sum(counter_M_improved)) 

                        self.N_MATCH = range(self.nr_matchings)
    
                        #print("Matching added.")
                        #if print_out:
                        #    print(found_M)

                        #if self.iterations == 10:
                        #    optimal = True
                        #    print("Process terminated after ", self.iterations, " iterations.")
                        self.iterations = self.iterations + 1    

                        # Remove constraint that reduced cost is negative again
                        self.pricing.constraints.pop("PricingObjConstr")            

                        
                        
                        # other problem to generate matchings
                        # The following function minimizes the average rank over all matchings with negative reduced cost.
                        #self.pricingMinRank(stab_constr, solver, print_log, time_limit, n_sol_pricingMinRank, gap_solutionpool_pricing, MIPGap, bool_ColumnGen, bool_supercolumn, print_out, pricing_obj, starting_time, solver_function)                        
                        
 
                    else:
                        optimal = True  
                        current_time = time.monotonic()
                        self.time_columnGen = current_time - starting_time
                        return self.generate_solution_report(print_out)    
                
                else:
                    # Infeasible, solve with smaller gap

                    if print_log == True:  
                        #self.pricing.solve(solver_function())
                        
                        #print("\n\n Careful, pricing stops at first postive value!\n\n")
                        #self.pricing.solve(solver_function(timeLimit = new_time_limit, BestObjStop = -constant +0.0001))
                        self.pricing.solve(solver_function(timeLimit = new_time_limit,
                        PoolGap = gap_solutionpool_pricing,
                        PoolSolutions = n_sol_pricing,
                        #BestObjStop = -constant +0.0001,
                        PoolSearchMode = 2, #Find diverse solutions
                        MIPGap = 0.001)) 

                        # Will stop the solver once a matching with objective function at least zero has been found
                        #self.pricing.solve(solver_function())

                    else:
                        #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
                        with open(os.devnull, 'w') as devnull:
                            with redirect_stdout(devnull):
                                #self.pricing.solve(solver_function(msg=False, timeLimit=new_time_limit, logPath = 'Logfile_pricing.log',BestObjStop = -constant + 0.0001))
                                #print("\n\n Careful, pricing stops at first postive value!\n\n")
                                self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log',timeLimit = new_time_limit,
                                    #BestObjStop = -constant +0.0001,
                                    PoolGap = gap_solutionpool_pricing,
                                    PoolSolutions = n_sol_pricing,
                                    PoolSearchMode = 2,
                                    MIPGap = 0.001))
                        #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
                    

                    # If not infeasible:
                    
                    if self.pricing.status not in [0,-1]: # -1 is infeasible, 0 is unsolved (for example because interrupted earlier, or timelimit reached)
                        #print("Status code: ", self.pricing.status)
                        obj_pricing_var = self.pricing.objective.value()
                        self.obj_pricing.append(obj_pricing_var)
                        if print_out:
                            print("self.obj_pricing_var: ", obj_pricing_var)
                            print("Constant term:", constant)
                            print("\t\tObjective pricing: ", obj_pricing_var)

                    #if print_out:
                    if False:
                        for (i,j) in self.PAIRS:
                            print("M[",i,j,'] =', self.M_pricing[i,j].value())
                    
                    #### EVALUATE SOLUTION ####
                    if print_out:
                        print('Pricing status', self.pricing.status)
                    if self.pricing.status == 0: # Time limit exceeded
                        self.time_limit_exceeded = True
                        optimal = False
                        self.time_columnGen = self.time_limit
                        return self.generate_solution_report(print_out) 
                    
                    elif self.pricing.status != -1:   
                        if obj_pricing_var > 0:
                            # Remove supercolumn if needed:
                            if self.supercolumn_in_model == True:
                                if remove_supercolumn == True:
                                    self.w[self.index_super_column].upBound = 0
                                    #self.master.writeLP("TestColumnFormulation2.lp")

                                    self.supercolumn_in_model = False


                            # The solution of the master problem is not optimal over all weakly stable matchings
                            
                            # Add non-negativity constraint to the master for this new matching
                            #name = 'GE0_' + str(len(self.w))
                            #self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
                            #self.master += self.constraints[name]

                            # Go through all generated solutions.
                            # To do this, first we should extract the gurobi model and access the solution pool through there.
                            pricing_gurobi = self.pricing.solverModel

                            n_sol_found = pricing_gurobi.SolCount

                            if print_out:
                                print('Solutions found by pricing:', n_sol_found)
                                #if stab_constr == "CUTOFF":
                                #    for i in self.SCHOOLS:
                                #        print("Cutoff school", i, ":", self.t[i].varValue)
                            
                            # Binary vector to remember which matchings were improved
                            counter_M_improved = np.zeros(n_sol_found)

                            # Go through all matchings
                            for t in tqdm(range(n_sol_found), desc='Found matchings added to master', unit='match', disable= not print_out):

                                pricing_gurobi.setParam('SolutionNumber', t)

                                # Add the matching found by the pricing problem to the master problem       
                                found_M = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

                                # Find all variables by name
                                for (i,j) in self.PAIRS:
                                    student_name = self.MyData.ID_stud[i]
                                    school_name = self.MyData.ID_school[j]
                                    name_var = f"M_{student_name}_{school_name}"
                                    gurobi_var = pricing_gurobi.getVarByName(name_var)
                                    found_M[i][j] = gurobi_var.Xn

                                self.M_list.append(found_M)
                                self.nr_matchings =self.nr_matchings + 1
                                self.N_MATCH = range(self.nr_matchings)
                                self.add_matching(found_M, len(self.w), print_out)
                                #if print_out:
                                #    print(found_M)
                                #self.master.writeLP("TestColumnFormulation.lp")
                            
                                
                                
                                # Exclude this matching from being find by the pricing problem in the future.
                                self.pricing += lpSum([self.M_pricing[i,j] * found_M[i][j] for (i,j) in self.PAIRS]) <= lpSum([found_M[i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{self.nr_matchings-1}"
                        
                                if print_out:  
                                    if t == 0:
                                        # Weirdly, the poolobjval does not take into account the constant value, 
                                        # while computing the objective value using pricing.objective.value() does?        
                                        #print("Constant value:", constant)
                                        #print("Pricing objective:", pricing_gurobi.PoolObjVal)
                                        #print("Computed objective value matching ", t, ": ", pricing_gurobi.PoolObjVal + constant)               
                                        print("Matching ", self.nr_matchings, "Objective value pricing:", pricing_gurobi.PoolObjVal + constant)

                                        
                                    elif t == n_sol_found - 1:
                                        print("Matching ", self.nr_matchings, "Objective value pricing:", pricing_gurobi.PoolObjVal + constant)

                            if print_out:
                                print("New number of matchings:", self.nr_matchings)
                                #print("Improving matchings found by SICs", sum(counter_M_improved)) 

                            self.N_MATCH = range(self.nr_matchings)

                            # Remove constraint that reduced cost is negative again
                            self.pricing.constraints.pop("PricingObjConstr")
        
                            #print("Matching added.")
                            #if print_out:
                            #    print(found_M)

                            #if self.iterations == 10:
                            #    optimal = True
                            #    print("Process terminated after ", self.iterations, " iterations.")
                            
                        else:
                            optimal = True
                            current_time = time.monotonic()
                            self.time_columnGen = current_time - starting_time
                            return self.generate_solution_report(print_out) 
                    else:
                        # Even with tighter gap, the pricing problem is still infeasible.
                        optimal = True
                        current_time = time.monotonic()
                        self.time_columnGen = current_time - starting_time
                        return self.generate_solution_report(print_out)     
            
            



            

    def build_pricing(self, stab_constr: str, print_out: bool):
        # Create Pulp model for pricing problem
        self.pricing = LpProblem("Pricing problem", LpMaximize)

        # Decision variables
        self.M_pricing = LpVariable.dicts("M", [(i, j) for i, j in self.PAIRS], cat="Binary")

        # Rename M
        for i, j in self.M_pricing:
            student_name = self.MyData.ID_stud[i]
            school_name = self.MyData.ID_school[j]

            self.M_pricing[i, j].name = f"M_{student_name}_{school_name}"

        ### CONSTRAINTS ###

        if stab_constr == 'TRAD':
            # Stability
            for i in self.STUD:
                for j in range(len(self.MyData.pref_index[i])):
                    current_school = self.MyData.pref_index[i][j]
                    lin = LpAffineExpression()

                    lin += self.MyData.cap[current_school] * self.M_pricing[i, current_school]

                    # Add all schools that are at least as preferred as the j-ranked school by student i
                    for l in range(j):
                        lin += self.MyData.cap[current_school] * self.M_pricing[i,self.MyData.pref_index[i][l]]


                    # Add terms based on priorities
                    prior_current = self.MyData.rank_prior[current_school][i]
                    for s in self.STUD:
                        if s != i:
                            # If current_school ranks student s higher than student i
                            if self.MyData.rank_prior[current_school][s] <= self.MyData.rank_prior[current_school][i]:
                                if (s, current_school) in self.PAIRS:
                                    lin += self.M_pricing[s,current_school]

                    # Add to model:
                    name = "STAB_" + str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_school[current_school]) 
                    self.pricing += (lin >= self.MyData.cap[current_school], name) 

        elif stab_constr == "CUTOFF":
            # Create decision variables cutoff scores, one for each school
            # NOTICE: The score of a student for a school, is the rank of the priority group to which they belong on that school
            # Contrary to the literature, all students with a LOWER score than the cutoff score will be admitted

            # NOTICE as well: We slightly modify non-envy from paper Agoston et al. (2022)
                # By removing the epsilon in constraints (6), we allow that some students from a priority group
                # are assigned, while others from the same priority group are not assigned because of capacities
                # This is the Irish system, and not the Hungarian or the Chilean (as they are called in that paper)
            
            self.t = LpVariable.dicts("t", [j for j in self.SCHOOLS], cat="Continuous")

            # Auxiliary parameter that contains number of priority groups at each school
            s_max = []
            for j in self.SCHOOLS:
                s_max.append(len(self.MyData.prior[j]))
            

            for (i,j) in self.PAIRS:
                # Find priority group to which i belongs at school j
                #s_i_j = s_max[j]
                #for k in range(s_max[j]):
                #    if isinstance(self.MyData.prior[j][k], tuple): # When more than a single student in this element
                #        if i in self.MyData.prior_index[j][k]:
                #            s_i_j = k

                # Rank of student i in priority of school j
                r_i_j = self.MyData.rank_prior[j][i]
                self.pricing += (self.t[j] >= self.M_pricing[(i,j)] * r_i_j, f"CUTOFF1_{i}_{j}")

                lin = LpAffineExpression()
                #current_school = self.MyData.pref_index[i][j]

                # Go through all schools that are weakly more preferred than current school by i
                for l in self.SCHOOLS:
                    if self.MyData.rank_pref[i][l] <= self.MyData.rank_pref[i][j]:
                        lin -= (self.M_pricing[(i,l)] * (s_max[j]+1))

                    
                # Add variable for cutoff score
                lin += self.t[j]

                # And add - score of student i at school j
                lin -= r_i_j

                # Add constraint to pricing model
                self.pricing += (lin <= 0, f"CUTOFF2_{i}_{j}")

            ######################################
            # Ensure non-wastefulness in pricing #
            ######################################
            
            # Add new binary variable for each school that is zero if capacity is not fully filled
            self.f = LpVariable.dicts("f", [j for j in self.SCHOOLS], cat="Binary")
            for j in self.SCHOOLS:
                lin = LpAffineExpression()
                for i in self.STUD:
                    if (i,j) in self.PAIRS:
                        lin += self.M_pricing[(i,j)]
                
                lin -= self.f[j] * self.MyData.cap[j]

                self.pricing += (lin >= 0, f"CUTOFF3_{j}")

            for j in self.SCHOOLS:
                lin = LpAffineExpression()
                self.pricing += (self.t[j] >= (1 - self.f[j]) * (s_max[j]+1), f"CUTOFF4_{j}")

        
        # Each student at most assigned to one school
        for i in self.STUD:
            self.pricing += lpSum([self.M_pricing[i,j] for j in self.SCHOOLS if (i,j) in self.PAIRS]) <= 1, f"LESS_ONE_{l}_{i}"

        # Capacities schools respected
        for j in self.SCHOOLS:
            self.pricing += lpSum([self.M_pricing[i,j] for i in self.STUD if (i,j) in self.PAIRS]) <= self.MyData.cap[j], f"LESS_CAP_{l}_{j}"
         
        # Exclude matchings that are already found:
        # Simple "no-good" cuts, where you sum matched student-school pairs for matching l, and force the sum to be strictly smaller
        # Required, because many matchings have same objective value in pricing problem,
            # and, sometimes, when a matching is added to the master and not immediately used, 
            # the dual prices are the same or similar, and the matching could have been found again by the pricing problem
        for l in tqdm(self.N_MATCH, desc='Pricing exclude found matchings', unit='matchings', disable=not print_out):            
            self.pricing += lpSum([self.M_pricing[i,j] * self.M_list[l][i][j] for (i,j) in self.PAIRS]) <= lpSum([self.M_list[l][i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{l}"
        
    def pricingMinRank(self, stab_constr: str, solver: str, print_log: str, time_limit: int, n_sol_pricing: int, gap_solutionpool_pricing: float, MIPGap: float, bool_ColumnGen: bool, bool_supercolumn: bool, print_out: bool, pricing_obj, starting_time, solver_function):
        # Also try another way to find interesting matchings
        if print_out:
            print("\n\n***************************\nFind matchings negative reduced cost with low rank\n***************************\n\n")
            #print(pricing_obj)

            print("\n\nATTENTION: dual modifier added to objective function!\n\n")
        
        # Add constraint to enforce that reduced cost is negative (or objective pricing positive)
        self.pricing += (pricing_obj >= 0, "PricingObjConstr") 

        self.pricing.sense = pl.LpMinimize
        # Change objective function to minimize rank
        new_obj = LpAffineExpression()
        for (i,j) in self.PAIRS:
            name_constr = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(j)
            dual_value=self.master.constraints[name_constr].pi
        
            new_obj += self.M_pricing[i,j]*(self.MyData.rank_pref[i,j]+ 1 +self.obj_master[-1] * dual_value/self.max_dual) / self.MyData.n_stud # + 1 because the indexing starts from zero
            #new_obj += self.M_pricing[i,j]*(self.MyData.rank_pref[i,j] + 1) / self.MyData.n_stud # + 1 because the indexing starts from zero
     
        
        # Add this variable to the model with the correct objective coefficient
        self.pricing.objective = LpAffineExpression() 
        self.pricing.setObjective(new_obj)
        #self.pricing.writeLP("PricingProblemMinRank.lp")

        # Find "n_solutions" best solutions of this formulation

        ###########
        # From here same code as for usual pricing problem (expect for returning the solution report (not possible because variables added to master by previous pricing))
        ###########

        # Update time limit:
        current_time = time.monotonic()
        new_time_limit = max(time_limit - (current_time - starting_time), 0)
        if print_out:
            print('New time limit', new_time_limit)

        if print_log == True:  
            #self.pricing.solve(solver_function())
            
            #print("\n\n Careful, pricing stops at first postive value!\n\n")
            #self.pricing.solve(solver_function(timeLimit = new_time_limit, BestObjStop = -constant +0.0001))
            self.pricing.solve(solver_function(timeLimit = new_time_limit,
            PoolGap = gap_solutionpool_pricing,
            PoolSolutions = n_sol_pricing,
            #BestObjStop = -constant +0.0001,
            PoolSearchMode = 2, #Find diverse solutions
            MIPGap = MIPGap)) 

            # Will stop the solver once a matching with objective function at least zero has been found
            #self.pricing.solve(solver_function())

        else:
            #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull):
                    #self.pricing.solve(solver_function(msg=False, timeLimit=new_time_limit, logPath = 'Logfile_pricing.log',BestObjStop = -constant + 0.0001))
                    #print("\n\n Careful, pricing stops at first postive value!\n\n")
                    self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log',timeLimit = new_time_limit,
                        #BestObjStop = -constant +0.0001,
                        PoolGap = gap_solutionpool_pricing,
                        PoolSolutions = n_sol_pricing,
                        PoolSearchMode = 2,
                        MIPGap = MIPGap))
            #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
        

        # If not infeasible:

        
        if self.pricing.status not in [0,-1]: # -1 is infeasible, 0 is unsolved (for example because interrupted earlier, or timelimit reached)
            #print("Status code: ", self.pricing.status)
            obj_pricing_var = self.pricing.objective.value()
            #self.obj_pricing.append(obj_pricing_var)
            if print_out:
                print("\t\tObjective pricing 2 (rank): ", obj_pricing_var)

        #if print_out:
        if False:
            for (i,j) in self.PAIRS:
                print("M[",i,j,'] =', self.M_pricing[i,j].value())
        
        #### EVALUATE SOLUTION ####
        if print_out:
            print('Pricing status', self.pricing.status)
        if self.pricing.status == 0: # Time limit exceeded
            self.time_limit_exceeded = True
            optimal = False
            self.time_columnGen = self.time_limit
            return 
                # We can't create solution report here, because matchings have been added in
                # previous pricing problem. Go back, and create solution report in main function
                    # based on check if pricing status == 0
            
        
        elif self.pricing.status != -1:   
            if obj_pricing_var > 0:
                # The solution of the master problem is not optimal over all weakly stable matchings
                
                # Add non-negativity constraint to the master for this new matching
                #name = 'GE0_' + str(len(self.w))
                #self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
                #self.master += self.constraints[name]

                # Go through all generated solutions.
                # To do this, first we should extract the gurobi model and access the solution pool through there.
                pricing_gurobi = self.pricing.solverModel

                n_sol_found = pricing_gurobi.SolCount

                if print_out:
                    print('Solutions found by pricing:', n_sol_found)
                
                # Binary vector to remember which matchings were improved
                counter_M_improved = np.zeros(n_sol_found)

                # Go through all matchings
                for t in range(n_sol_found):

                    pricing_gurobi.setParam('SolutionNumber', t)

                    # Add the matching found by the pricing problem to the master problem       
                    found_M = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

                    # Find all variables by name
                    for (i,j) in self.PAIRS:
                        student_name = self.MyData.ID_stud[i]
                        school_name = self.MyData.ID_school[j]
                        name_var = f"M_{student_name}_{school_name}"
                        gurobi_var = pricing_gurobi.getVarByName(name_var)
                        found_M[i][j] = gurobi_var.Xn


                    self.M_list.append(found_M)
                    self.nr_matchings = self.nr_matchings + 1
                    self.N_MATCH = range(self.nr_matchings)
                    self.add_matching(found_M, len(self.w), print_out)
                    #self.master.writeLP("TestColumnFormulation.lp")

                    # Test stability of matching
                    #is_stable = stability_test_single_matching(self.MyData, found_M, print_out)
                    #if not is_stable:
                    #    print(found_M)
                    #    # If not stable, print values
                    #    if self.stab_constr == "CUTOFF":
                    #        if print_out:
                    #            print("Matching found by pricing is not stable! Cutoffs:")
                    #        for i in self.SCHOOLS:
                    #            if print_out:
                    #                print("Cutoff school", i, ":", self.t[i].varValue)
                    #        for i in self.SCHOOLS:
                    #            if print_out:
                    #                print("Capacity filled school", i, ":", self.f[i].varValue)
                    #    raise Exception("Matching found by pricing problem is not stable!")
                
                    
                    
                    # Exclude this matching from being find by the pricing problem in the future.
                    self.pricing += lpSum([self.M_pricing[i,j] * found_M[i][j] for (i,j) in self.PAIRS]) <= lpSum([found_M[i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{self.nr_matchings-1}"
            
                    if print_out:  
                        if t == 0:               
                            print("Matching ", self.nr_matchings, "Objective value pricing 2 (rank):", pricing_gurobi.PoolObjVal)

                            
                        elif t == n_sol_found - 1:
                            print("Matching ", self.nr_matchings, "Objective value pricing 2 (rank):", pricing_gurobi.PoolObjVal)

                    # Find SICs
                    #M_SIC = SIC(self.MyData, found_M, False)
                    #if not np.array_equal(M_SIC, found_M): # If improvement realized by executing SICs
                    #    counter_M_improved[t] = 1
                    #    self.M_list.append(M_SIC)
                    #    self.nr_matchings = self.nr_matchings + 1
                    #    self.N_MATCH = range(self.nr_matchings)
                    #    self.add_matching(M_SIC, len(self.w), print_out)
                    #    #self.master.writeLP("TestColumnFormulation.lp")
                                        #    self.M_list.append(M_SIC)
                        
                       
                    #    # Exclude this matching from being find by the pricing problem in the future.
                    #    self.pricing += lpSum([self.M_pricing[i,j] * M_SIC[i][j] for (i,j) in self.PAIRS]) <= lpSum([M_SIC[i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{self.nr_matchings-1}"
                #if print_out:
                #    print("Improving matchings found by SICs", sum(counter_M_improved))

        # Remove constraint that reduced cost is negative again
        self.pricing.constraints.pop("PricingObjConstr")
        
        # Change back to maximization
        self.pricing.sense = pl.LpMaximize

    def print_dual_values(self):
        # Get dual variables
        duals = {}
        
        duals["Sum_to_one"] = self.master.constraints["Sum_to_one"].pi

        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                school_name = self.MyData.pref_index[i][j]
                name_duals = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                name_constr = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                    
                duals[name_duals]=self.master.constraints[name_constr].pi
                
                if abs(duals[name_duals] < 0.00001): # Get rid of small numerical inaccuracies
                    duals[name_duals] = 0

                
                if duals[name_duals] > 0.00001:
                    print("Dual ", name_duals, ":", duals[name_duals])  

            
        constant = 0
        constant += duals['Sum_to_one']

        print("Constant term", constant)

        

    def get_pricing_objective_of_matching(self, M:np.ndarray, print_out:bool):
        pricing_obj_value = 0
        
        # Get dual variables
        duals = {}
        
        duals["Sum_to_one"] = self.master.constraints["Sum_to_one"].pi

        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                school_name = self.MyData.pref_index[i][j]
                name_duals = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                name_constr = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                    
                duals[name_duals]=self.master.constraints[name_constr].pi
                
                if abs(duals[name_duals] < 0.00001): # Get rid of small numerical inaccuracies
                    duals[name_duals] = 0

                if print_out:
                    if duals[name_duals] > 0.00001:
                        print("Dual ", name_duals, ":", duals[name_duals])  

                
            
        # Modify objective function pricing problem
        for i in self.STUD:
            #print('student ', i)
            for j in range(len(self.MyData.pref[i])): 
                school_name = self.MyData.pref_index[i][j]
                if self.bool_punish_unassigned == False:
                    pricing_obj_value += M[i,school_name] * ( - (self.MyData.rank_pref[i,school_name]+ 1) / self.MyData.n_stud ) # + 1 because the indexing starts from zero
                else:   
                    raise NotImplementedError(f"The pricing problem with punishing unassigned students is not yet implemented. Please set bool_punish_unassigned to False.")
                
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    name = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(pref_school)
                    pricing_obj_value += M[i,pref_school] * duals[name] 


        

        if self.bool_identical_students:
            raise NotImplementedError("Getting pricing objective of a matching not implemented for identical students case yet.")
            


        #if print_out:
        #    print(pricing_obj)
        #print("Duals Sum_to_one", duals["Sum_to_one"])
        #print("Duals", duals)

        # Add the constant terms!
        constant = 0
        constant += duals['Sum_to_one']
        pricing_obj_value += constant

        return pricing_obj_value


    def add_n_new_matchings(self, n_generated: int, seed:int, print_out: bool):

        A_extra = DA_STB(self.MyData, n_generated, 'EE', False, seed, print_out)

        # Find Stable improvement cycles à la Erdil and Ergin (2008)
        A_SIC_extra = SIC_all_matchings(self.MyData, A_extra, print_out)
        A_SIC_M_list =np.array(list(A_SIC_extra.M_set))

        # Add matchings in A_SIC_extra to model
        for m in tqdm(range(len(A_SIC_M_list)), desc='Master: add decision variables', unit='var', disable= not print_out):
            #print(self.M[m])
            self.M_list.append(A_SIC_M_list[m])
            self.nr_matchings =self.nr_matchings + 1
            self.N_MATCH = range(self.nr_matchings)
            self.add_matching(A_SIC_M_list[m], len(self.w), False)


    def generate_solution_report(self, print_out = False):
        # Store everything in a solution report
        S = SolutionReport()

        # Check if supercolumn is still in model. If yes, check if it has weight zero. If not, the problem is infeasible
        if self.supercolumn_in_model == True:
            if self.w[self.index_super_column].varValue > 0.00001:
                if print_out:
                    print("Supercolumn has positive weight in final solution, master problem is infeasible!")
                self.master.status = -1 # Infeasible

        if self.master.status == -1: # If the master problem is infeasible (e.g., because we can't sd-dominate EADA)
            S.optimal = False
            S.time_limit_exceeded = False
            S.time = self.time_columnGen
            if print_out:
                print("INFEASIBLE master problem, could not find an ex-post stable assignment that sd-dominates given assignment.\n")

        elif self.bool_ColumnGen == False:
            S.optimal = False
            S.time_limit_exceeded = False
            S.time = self.time_columnGen
            if print_out:
                print("Average rank of heuristic: \t", self.obj_master[-1])
        
        elif self.time_limit_exceeded == False:  
            # Optimal solution of master problem!
            S.optimal = True
            S.time_limit_exceeded = False
            S.time = self.time_columnGen
            S.time_limit = self.time_limit
            if print_out:
                print("Optimal solution found!\nBest average rank: \t", self.obj_master[-1])

        else:
            # Time limit exceeded
            S.optimal = False
            S.time_limit_exceeded = True
            S.time = self.time_limit
            S.time_limit = self.time_limit

            if print_out:
                print('\nTime limit of ', self.time_limit, "seconds exceeded!\n")
                if self.bool_ColumnGen == True:
                    print('Rank best found solution:\t', self.obj_master[-1])
        
        S.avg_ranks = {}

        if self.master.status == -1: # If master problem infeasible
            S.avg_ranks['result'] = None
            S.avg_ranks['first_iter']  = None
            S.avg_ranks['warm_start'] = self.avg_rank
            S.avg_ranks['DA'] = self.avg_rank_DA

            # Save the final solution
            # Create variables to store the solution in
            
            zero = np.full((self.MyData.n_stud, self.MyData.n_schools), np.nan)
            self.Xassignment = Assignment(self.MyData, zero) # Contains the final assignment found by the model
            
            # Make sure assignment is empty in Xassignment
            self.Xassignment.assignment = np.full((self.MyData.n_stud, self.MyData.n_schools), np.nan)
            S.obj_master = []
            n_match_support = 0
            # Store decomposition
            #self.Xdecomp = [] # Matchings in the found decomposition
            #self.Xdecomp_coeff = [] # Weights of these matchings

        else : # If NOT infeasible
            S.avg_ranks['result'] = self.obj_master[-1]
            S.avg_ranks['first_iter']  = self.avg_rank_first_iter
            S.avg_ranks['warm_start'] = self.avg_rank
            S.avg_ranks['DA'] = self.avg_rank_DA

            if print_out:
                print("Rank first iteration: \t", self.avg_rank_first_iter)
                print("Rank warm start solution: \t", self.avg_rank)
                print("Original average rank: \t", self.avg_rank_DA)

            S.obj_master = self.obj_master
            S.obj_pricing = self.obj_pricing

            # Save the final solution
            # Create variables to store the solution in
            
            zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
            self.Xassignment = Assignment(self.MyData, zero) # Contains the final assignment found by the model
            
            # Make sure assignment is empty in Xassignment
            self.Xassignment.assignment = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

            # Store decomposition
            #self.Xdecomp = [] # Matchings in the found decomposition
            #self.Xdecomp_coeff = [] # Weights of these matchings

            #if print_out:
                #print('self.N_MATCH', self.N_MATCH)
                #print('self.w', len(self.w))
                #print('self.M_list', len(self.M_list))
            n_match_support = 0
            for l in self.N_MATCH:
                #self.Xdecomp.append(np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools)))
                #self.Xdecomp_coeff.append(self.w[l].varValue)
                for (i,j) in self.PAIRS:
                    #self.Xdecomp[-1][i,j] = self.M_list[l][i][j]
                    self.Xassignment.assignment[i,j] += self.w[l].varValue * self.M_list[l][i][j]
                if self.w[l].varValue > 0.00001:
                    n_match_support += 1
                
            
        #S.Xdecomp = self.Xdecomp
        #S.Xdecomp_coeff = self.Xdecomp_coeff
        S.A = copy.deepcopy(self.Xassignment)

        S.n_students_assigned = S.A.compute_n_assigned_students()
        S.n_match_support = n_match_support

        S.A_SIC = copy.deepcopy(self.p)
        S.A_DA_prob = copy.deepcopy(self.p_DA)

        S.iter = self.iterations
        #S.n_match = len(self.N_MATCH)
        S.n_match = len(self.M_list)

        S.bool_ColumnGen = self.bool_ColumnGen

        return S
