from .Assignment import *

# Install necessary packages before running the script:
# pip install pulp
# pip install gurobipy
# pip install pyscipopt
# Note: Obtain an academic license for Gurobi from: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

# Check with solvers available on computer
import pulp as pl
from pulp import *

import gurobipy

# To check which solvers available on computer:
# print(pl.listSolvers(onlyAvailable=True))

class Model: 
    """
    Solves the large ILP formulation in the paper to find a random matching that is ex-post stable and sd-dominates a given random matching.
    Contains two methods:
        __init__: initializes the model, and the solver environment

        Solve: solves the model.
            The parameters of this method can control which objective function is optimized, and which solver is used
    """
    
    # Used this example as a template for Pulp: https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
    
    def __init__(self, MyData: Data, p: Assignment, print_out: bool, stab_constr: str, nr_matchings = -1):
        """
        Initialize an instance of Model.

        Args:
            MyData (type: Data): instance of class Data.
            p (type: Assignment): instance of class Assignment.
            print_out (type: bool): boolean that controls which output is printed.
            stab_constr (str): controls which type of stability constraints are used.
            nr_matchings (optional): number of matchings used in the decomposition, optional parameter that defaults to n_students * n_schools + 1

        """
        # Valid values for 'stab_constr'
        stab_list = ["TRAD", "CUTOFF"]
        if stab_constr not in stab_list:
           raise ValueError(f"Invalid value: '{stab_constr}'. Allowed values are: {stab_list}")    

        # 'nr_matchings' refers to number of matchings used to find decomposition
        self.MyData = copy.deepcopy(MyData)
        self.p = copy.deepcopy(p)
        self.nr_matchings = nr_matchings
        if nr_matchings == -1:
            self.nr_matchings = self.MyData.n_stud * self.MyData.n_schools + 1

        # Create the pulp model
        self.model = LpProblem("Improving_ex_post_stable_matchings", LpMinimize)

        # Create variables to store the solution in
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings
        zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
        self.Xassignment = Assignment(MyData, zero) # Contains the final assignment found by the model

        #### DECISION VARIABLES ####
        self.STUD = range(0,self.MyData.n_stud)
        self.SCHOOLS = range(0, self.MyData.n_schools)
        self.N_MATCH = range(0, self.nr_matchings)

        # Tuple with all student-school pairs that are preferred to outside option
        # This tuple contains the INDICES of the students and the pairs, and not their original names
        self.PAIRS = []
        for i in range(0, MyData.n_stud):
            for j in range(0,len(MyData.pref[i])):
                # Convert pref[i][k] (school ID as string) to column index
                col_index = MyData.ID_school.index(MyData.pref[i][j])
                self.PAIRS.append((i,col_index))   
        
        # M[k][i][j] = 1 if student i is assigned to school j in matching k, and 0 otherwise
        self.M = LpVariable.dicts("M", [(k, i, j) for k in self.N_MATCH for i, j in self.PAIRS], cat="Binary")

        # Auxiliary variables to avoid non-linearity
        self.z = LpVariable.dicts("z", [(k, i, j) for k in self.N_MATCH for (i, j) in self.PAIRS], 0, 1)

        # Rename M and z
        for k, i, j in self.M:
            student_name = self.MyData.ID_stud[i]
            school_name = self.MyData.ID_school[j]
            self.M[k, i, j].name = f"M_{k}_{student_name}_{school_name}"
            self.z[k, i, j].name = f"z_{k}_{student_name}_{school_name}"

        # Q[i][j] is the new probability with which student i is assigned to school j, lies between 0 and 1
        self.Q = LpVariable.dicts("q", self.PAIRS, 0, 1) 
    
        # w[k] is the weight of matching k in the decomposition
        self.w = LpVariable.dicts("w", self.N_MATCH, 0, 1)

        #### OBJECTIVE FUNCTION ####
            # Done separately in other functions (see function Solve)
        
            
        #### CONSTRAINTS ####
        # Other constraints defined for specific models in functions below (see function Solve)

        # Stability
        if stab_constr == "TRAD":
            for k in self.N_MATCH:
                for i in self.STUD:
                    for j in range(len(self.MyData.pref_index[i])):
                        current_school = self.MyData.pref_index[i][j]
                        lin = LpAffineExpression()

                        lin += self.MyData.cap[current_school] * self.M[k, i, current_school]

                        # Add all schools that are at least as preferred as the j-ranked school by student i
                        for l in range(j):
                            lin += self.MyData.cap[current_school] * self.M[k,i,self.MyData.pref_index[i][l]]


                        # Add terms based on priorities
                        prior_current = self.MyData.rank_prior[current_school][i]
                        for s in self.STUD:
                            if s != i:
                                # If current_school ranks student s higher than student i
                                if self.MyData.rank_prior[current_school][s] <= self.MyData.rank_prior[current_school][i]:
                                    if (s, current_school) in self.PAIRS:
                                        lin += self.M[k,s,current_school]

                        # Add to model:
                        name = "STAB_" + str(k) + "_" + str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_school[current_school]) 
                        self.model += (lin >= self.MyData.cap[current_school], name) 

        elif stab_constr == "CUTOFF":
            # Create decision variables containing cutoff ranks
            self.t = LpVariable.dicts("t", [(k,j) for k in self.N_MATCH for j in self.SCHOOLS], cat="Continuous")
            
            # Auxiliary parameter that contains number of priority groups at each school
            s_max = []

            # Add new binary variable for each school that is zero if capacity is not fully filled
            self.f = LpVariable.dicts("f", [(k,j) for k in self.N_MATCH for j in self.SCHOOLS], cat="Binary")

            for j in self.SCHOOLS:
                s_max.append(len(self.MyData.prior[j]))

            for k in self.N_MATCH:
                for (i,j) in self.PAIRS:
                    # Find priority group to which i belongs at school j
                    #s_i_j = s_max[j]
                    #for k in range(s_max[j]):
                    #    if isinstance(self.MyData.prior[j][k], tuple): # When more than a single student in this element
                    #        if i in self.MyData.prior_index[j][k]:
                    #            s_i_j = k

                    # Rank of student i in priority of school j
                    r_i_j = self.MyData.rank_prior[j][i]
                    self.model += (self.t[k,j] >= self.M[(k,i,j)] * r_i_j, f"CUTOFF1_{k}_{i}_{j}")

                    lin = LpAffineExpression()
                    current_school = self.MyData.pref_index[i][j]

                    # Go through all schools that are weakly more preferred than current school by i
                    for l in self.SCHOOLS:
                        if self.MyData.rank_pref[i][l] <= self.MyData.rank_pref[i][j]:
                            lin -= (self.M[(k,i,l)] * (s_max[j]+1))

                        
                    # Add variable for cutoff score
                    lin += self.t[k,j]

                    # And add - score of student i at school j
                    lin -= r_i_j

                    # Add constraint to model
                    self.model += (lin <= 0, f"CUTOFF2_{k}_{i}_{j}")

                ######################################
                # Ensure non-wastefulness in model #
                ######################################
                

                for j in self.SCHOOLS:
                    lin = LpAffineExpression()
                    for i in self.STUD:
                        if (i,j) in self.PAIRS:
                            lin += self.M[(k,i,j)]
                    
                    lin -= self.f[k,j] * self.MyData.cap[j]

                    self.model += (lin >= 0, f"CUTOFF3_{k}_{j}")

                for j in self.SCHOOLS:
                    lin = LpAffineExpression()
                    self.model += (self.t[k,j] >= (1 - self.f[k,j]) * (s_max[j]+1), f"CUTOFF4_{k}_{j}")

        
        # Each student at most assigned to one school
        for l in self.N_MATCH:
            for i in self.STUD:
                self.model += lpSum([self.M[l,i,j] for j in self.SCHOOLS if (i,j) in self.PAIRS]) <= 1, f"LESS_ONE_{l,i}"

        # Capacities schools respected
        for l in self.N_MATCH:
            for j in self.SCHOOLS:
                self.model += lpSum([self.M[l,i,j] for i in self.STUD if (i,j) in self.PAIRS]) <= self.MyData.cap[j], f"LESS_CAP_{l,j}"

        # Symmetry breaking
        #for l in range(self.nr_matchings-1):
        #    self.model += (self.w[l] >= self.w[l+1], f"SYMM_{l}")

        for l in range(self.nr_matchings-1):
            lin = LpAffineExpression()

            for j in self.SCHOOLS:
                lin += (j+1) * self.M[l,i,j]
                lin -= (j+1) * self.M[l+1, i,j]
            self.model += (lin <= 0, f"SYMM_{l}")



    def Solve(self, obj: str, solver: str, print_out: bool):
        """
        Solves the formulation.
        Returns an instance from the Assignment class.

        Args:
            obj (str): controls the objective function
                "IMPR_RANK": minimizes expected rank while maintaining ex-post stability
                "STABLE": maximizes fraction of stable matchings in decomposition
                "EX_ANTE": finds ex-ante stable improvement (heuristic)
            solver (str): controls which solver is used. See options through following commands:
                solver_list = pl.listSolvers(onlyAvailable=True)
                print(solver_list)
            print_out (bool): boolean that controls which output is printed.

        """

        # Check that strings-arguments are valid

        # Valid values for 'solver'
        solver_list = pl.listSolvers(onlyAvailable=True)
        if solver not in solver_list:
           raise ValueError(f"Invalid value: '{solver}'. Allowed values are: {solver_list}")

        # Valid values for 'obj'
        obj_list = ["IMPR_RANK", "STABLE", "PENALTY"]
        if obj not in obj_list:
           raise ValueError(f"Invalid value: '{obj}'. Allowed values are: {obj_list}")

        #### FORMULATION ####
        
        # Set the objective function
        if obj == "IMPR_RANK":
            self.Improve_rank(print_out)
        
        elif obj == "STABLE":
            self.Max_Stable_Fraction(print_out)

        elif obj == "PENALTY":
            self.Improve_rank_penalty(print_out)
            # Find decomposition with supercolumn that tries to find decomposition with only stable matching

        self.model.writeLP("TestFormulation.lp")

        
        #### SOLVE ####
            
        # String can't be used as the argument in solve method, so convert it like this:
        solver_function = globals()[solver]  # Retrieves the GUROBI function or class
        
        # Solve the formulation
        self.model.solve(solver_function())
        #self.model.solve(GUROBI_CMD(keepFiles=True, msg=True, options=[("IISFind", 1)]))
        
        #### STORE SOLUTION ####
        # Make sure assignment is empty in Xassignment
        self.Xassignment.assignment = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

        for (i,j) in self.PAIRS:
            self.Xassignment.assignment[i,j] = self.Q[i,j].varValue

        # Store decomposition
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings

        for l in self.N_MATCH:
            self.Xdecomp.append(np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools)))
            self.Xdecomp_coeff.append(self.w[l].varValue)
            for (i,j) in self.PAIRS:
                self.Xdecomp[-1][i,j] = self.M[l,i,j].varValue
                
        return self.Xassignment


    def Improve_rank(self, print_out: bool):
        """
        Creates and solves formulation to minimize the expected rank while ensuring the found random matching is ex-post stable.
        """
        
        if print_out == True:
            # Compute average rank of current assignment

            sum = 0
            for (i,j) in self.PAIRS:
                sum += self.p.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
            # Average
            sum = sum/self.MyData.n_stud
            print(f"\nAverage rank before optimization: {sum}.\n\n")
        
        # Objective function
        lin = LpAffineExpression()
        for (i,j) in self.PAIRS:
            lin += (self.Q[i,j] * (self.MyData.rank_pref[i,j] + 1)) / self.MyData.n_stud # + 1 because the indexing starts from zero
        self.model += lin

        # Define q based on matchings in decomposition
            # Where z is an auxiliary variable to avoid non-linearities
        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.w[l] <= 0,f"z_w{l,i,j}" 

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.M[l, i, j] <= 0,f"z_M_{l, i, j}"

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] + (1 - self.M[l, i, j]) - self.w[l]  >= 0,f"z_w_M_{l, i, j}"
                # Maybe these constraints are redundant because of the objective function

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) == self.Q[i,j], f"z_Q_{i, j}"

        # Ensure weights sum up to one
        self.model += lpSum([self.w[l] for l in self.N_MATCH]) == 1, f"SUM_TO_ONE"

        # First-order stochastic dominance
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                lin = LpAffineExpression()
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    lin += self.Q[i,pref_school]
                    lin -= self.p.assignment[i,pref_school]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(j)
                self.model += (lin >= 0, name)


    def Max_Stable_Fraction(self, print_out: bool):
        # Objective function
        obj = LpAffineExpression()
        for l in self.N_MATCH:
            obj += self.w[l] 
        self.model += obj
        self.model.sense = LpMaximize

        # Constraints to ensure that decomposition is at least equal to p (element-wise)
            # Where z is an auxiliary variable to avoid non-linearities
        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.w[l] <= 0,f"z_w{l,i,j}" 

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.M[l, i, j] <= 0,f"z_M_{l, i, j}"

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] + (1 - self.M[l, i, j]) - self.w[l]  >= 0,f"z_w_M_{l, i, j}"
                # Maybe these constraints are redundant because of the objective function

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) == self.Q[i,j], f"z_Q_{i, j}"

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) <= self.p.assignment[i,j], f"z_p_{i, j}"


    def Improve_rank_penalty(self, print_out:bool):
        if print_out == True:
            # Compute average rank of current assignment

            sum = 0
            for (i,j) in self.PAIRS:
                sum += self.p.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
            # Average
            sum = sum/self.MyData.n_stud
            print(f"\nAverage rank before optimization: {sum}.\n\n")
        
        self.W = LpVariable("W", 0, 1) # The weight of the supercolumn

        # Objective function
        lin = LpAffineExpression()
        for (i,j) in self.PAIRS:
            lin += (self.Q[i,j] * (self.MyData.rank_pref[i,j] + 1)) / self.MyData.n_stud # + 1 because the indexing starts from zero
        
        # Add large penalty for supercolum. For example, if selected with 0.0001, then should still be larger than #schools
        lin += self.W * 10000*self.MyData.n_schools
        self.model += lin

        # Define q based on matchings in decomposition
            # Where z is an auxiliary variable to avoid non-linearities
        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.w[l] <= 0,f"z_w{l,i,j}" 

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.M[l, i, j] <= 0,f"z_M_{l, i, j}"

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] + (1 - self.M[l, i, j]) - self.w[l]  >= 0,f"z_w_M_{l, i, j}"
                # Maybe these constraints are redundant because of the objective function

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) + self.W == self.Q[i,j], f"z_Q_{i, j}"

        # Ensure weights sum up to one
        self.model += lpSum([self.w[l] for l in self.N_MATCH]) + self.W == 1, f"SUM_TO_ONE"

        # First-order stochastic dominance
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                lin = LpAffineExpression()
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    lin += self.Q[i,pref_school]
                    lin -= self.p.assignment[i,pref_school]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(j)
                self.model += (lin >= 0, name)




    def print_solution(self):
        s = "The obtained random matching is:\n"
        s+=f"\t\t"
        for j in self.SCHOOLS:
            s+=f"{self.MyData.ID_school[j]}\t"
        s+="\n"
        for i in self.STUD:
            s+= f"\t{self.MyData.ID_stud[i]}\t"
            for j in self.SCHOOLS:
                s+=f"{self.Xassignment.assignment[i,j]}\t"
            s+=f"\n"
        s+=f"\n"

        s+= "The matchings with positive weights are:\n"

        for l in self.N_MATCH:
            if self.Xdecomp_coeff[l] > 0:
                s+=f"\t w[{l}] = {self.Xdecomp_coeff[l]}\n"
                for i in self.STUD:
                    s+=f"\t\t"
                    for j in self.SCHOOLS:
                        if self.Xdecomp[l][i,j] == 1:
                            s+=f"1\t"
                        else:
                            s+= f"0\t"
                    s+=f"\n"
                s+=f"\n"
        print(s)


    def WarmStart(self, print_out:bool):
        # Find a warm start based on the matchings that were found to sample DA,
        # and the matchings where the SICs were resolved.

        # Create the pulp model
        self.WarmSModel = LpProblem("WarmStart_Improving_ex_post_stable_matchings", LpMinimize)

        
                
        