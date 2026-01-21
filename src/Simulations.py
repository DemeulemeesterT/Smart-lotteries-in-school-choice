from src.Data import * # Class containing the data
from src.Assignment import * # Class containing an assignment
from src.Model import * # Class containing a Pulp model used for optimization
from src.ModelColumnGen import * # Class containing Pulp model that optimizes using column generation
from src.DataGen import * # Generate student preferences and school priorities
from src.DataGenEE import * # Generate data according to the method by Erdil & Ergin (2008)
from src.DA_STB import * # Generate DA assignment with single tie-breaking (STB)
from src.ErdilErgin import * # Erdil & Ergil their implementation of Stable Improvement Cycles algorithm + alternative implementation DA
from src.SICs import * # Adaptation of SICs algorithm to our code
from src.EADAM import * # EADA algorithm



def SimulationCG(COMPARE_SOLUTIONS: list, n_students_schools: list, alpha: list, beta: list, n_iterations_simul: int, n_match: int, n_sol_extra_sample: int, time_lim: int, seed: int, n_sol_pricing: int, n_sol_pricingMinRank:int, gap_pricing: float, MIPGap: float, bool_ColumnGen: bool, bool_supercolumn: bool, bool_identical_students: bool, print_out = False):
    """
    Will run column generation framework 'n_iterations' times, for the specified parameter values
    Output: an array containing SolutionReport objects

    - COMPARE_SOLUTIONS: a list of strings that contains the solutions we want to compare
    - n_students_schools, alpha, beta: lists containing the corresponding parameter values
    - n_students_schools is a list of pairs of values to evaluate: e,g,. [[100, 2], [200, 4], [400, 8]]
    - n_sol_extra_sample: number of additional weakly stable matchings sampled when evaluating nethods with larger initial sample sizes

    For column generation
    - n_match: number of solutions returned by pricing problem
    - n_sol_pricingMinRank: number of solutions returned by the pricing problem that minimizes the average rank
    - gap_pricing: optimality gap used for the solutions included in the solution pool in the pricing problem
    - MIPGap: gap for pricing problem
    - bool_ColumnGen: if True: perform entire column generation for time_limit period
                      if False: only perform first iteration, and don't build pricing problem
    - bool_supercolumn: boolean to determine whether you want to add an artificial column to the model when the model is infeasible?
    - bool_identical_students (bool): if True, give identical students the same probabilities

    For each combination of parameter values (n_stud, n_school, alpha, beta), we start seed from same value
    This makes it easier to reproduce the results later on
    """
    print_intermediate = print_out
    print_counter = True

    # Create directory if it doesn't exist
    os.makedirs('Simulation Results', exist_ok=True)

    S_vector = []

    now = time.strftime('%Y-%m-%d_%H%M%S')

    # The different possible solutions
    sol_list = ["SD_UPON_DA", "SD_UPON_EE", "SD_UPON_EADA", "SD_UPON_DA_SAMPLE", "SD_UPON_EE_SAMPLE", "SD_UPON_EADA_SAMPLE"]
    for sol in COMPARE_SOLUTIONS:
        if sol not in sol_list:
            raise ValueError(f"Invalid value: '{sol}'. Allowed values are: {sol_list}")   
        
    # Create csv file
    output_file = 'Simulation Results/SIM_' + now + '.csv'
    headers = ['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'time_limit', '#_sol_methods', 'avg_rank_DA', 'avg_rank_EE', 'avg_rank_EADA', 'n_assigned_DA', 'n_assigned_EE', 'n_assigned_EADA']
    counter = 1
    for i in COMPARE_SOLUTIONS:
        label_name = f'sol_{counter}_label'
        headers.append(label_name)
        counter = counter + 1
    counter = 1
    for i in COMPARE_SOLUTIONS: # For each solution method, we store all the following performance measures
        label_name = f'{counter}_avg_rank_result'
        headers.append(label_name)
        label_name = f'{counter}_avg_rank_heur'
        headers.append(label_name)
        label_name = f'{counter}_n_assigned'
        headers.append(label_name)
        label_name = f'{counter}_bool_CG'
        headers.append(label_name)
        label_name = f'{counter}_time'
        headers.append(label_name)
        label_name = f'{counter}_time_lim_exceeded'
        headers.append(label_name)
        label_name = f'{counter}_optimal'
        headers.append(label_name)
        label_name = f'{counter}_n_iter'
        headers.append(label_name)
        label_name = f'{counter}_n_match'
        headers.append(label_name)
        label_name = f'{counter}_n_match_support'
        headers.append(label_name)
        counter = counter + 1

    counter = 1
    for i in COMPARE_SOLUTIONS:
        # And some other performance metrics
        label_name = f'{counter}_n_stud_impr_DA'
        headers.append(label_name)
        label_name = f'{counter}_avg_rank_impr_DA'
        headers.append(label_name)
        label_name = f'{counter}_median_rank_impr_DA'
        headers.append(label_name)
        label_name = f'{counter}_n_stud_impr_EE'
        headers.append(label_name)
        label_name = f'{counter}_avg_rank_impr_EE'
        headers.append(label_name)
        label_name = f'{counter}_median_rank_impr_EE'
        headers.append(label_name)

        if ("SD_UPON_EADA" in COMPARE_SOLUTIONS) or ("SD_UPON_EADA_SAMPLE" in COMPARE_SOLUTIONS):
            label_name = f'{counter}_n_stud_impr_EADA'
            headers.append(label_name)
            label_name = f'{counter}_avg_rank_impr_EADA'
            headers.append(label_name)
            label_name = f'{counter}_median_rank_impr_EADA'
            headers.append(label_name)
        counter = counter + 1

    # Add comparisons between the number of students who improve in the base cases (e.g., how many students improve in EE compared to DA)
    label_name = f'n_stud_impr_EE_DA'
    headers.append(label_name)
    label_name = f'avg_rank_impr_EE_DA'
    headers.append(label_name)
    label_name = f'median_rank_impr_EE_DA'
    headers.append(label_name)

    if ("SD_UPON_EADA" in COMPARE_SOLUTIONS) or ("SD_UPON_EADA_SAMPLE" in COMPARE_SOLUTIONS):
        label_name = f'n_stud_impr_EADA_DA'
        headers.append(label_name)
        label_name = f'avg_rank_impr_EADA_DA'
        headers.append(label_name)
        label_name = f'median_rank_impr_EADA_DA'
        headers.append(label_name)

        label_name = f'n_stud_impr_EADA_EE'
        headers.append(label_name)
        label_name = f'avg_rank_impr_EADA_EE'
        headers.append(label_name)
        label_name = f'median_rank_impr_EADA_EE'
        headers.append(label_name)

        label_name = f'n_stud_impr_EE_EADA'
        headers.append(label_name)
        label_name = f'avg_rank_impr_EE_EADA'
        headers.append(label_name)
        label_name = f'median_rank_impr_EE_EADA'
        headers.append(label_name)

    
    # Create csv in which final results will be stored
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader() # Writes the header row from 'headers

    # Create pandas dataframe to store results along the way
    df = pd.DataFrame(columns=headers)

    # Second csv file that stores values of master objective
    # Create csv file
    output_file_obj = 'Simulation Results/SIM_' + now + '_obj.csv'
    headers2 = ['n_stud', 'n_schools', 'alpha', 'beta', 'seed', "sol_method", "iterations", "master objective values"]

    with open(output_file_obj, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers2)
        writer.writeheader() # Writes the header row from 'headers
    

    total_combinations = len(n_students_schools) * len(alpha) * len(beta)

    for k, b, a in tqdm(itertools.product(n_students_schools, beta, alpha), total = total_combinations, desc='Data instances', unit = 'inst', disable= not print_counter):
        # Generate data using data generation by Erdil & Ergin (2008)
        random.seed(seed)

        n = k[0] # Number of students
        m = k[1] # Number of schools

        seed_vector = []
        for i in range(n_iterations_simul):
            seed_vector.append(random.randint(0,1000000000))

        for i in tqdm(range(n_iterations_simul), desc = 'iterations', total = n_iterations_simul, unit = 'iter', disable = not print_out):
            if print_out:
                print('\nn,m,alpha, beta, seed', n, m, a, b, seed_vector[i])
            
            row_data = {
                'n_stud' : n,
                'n_schools' : m,
                'alpha' : a,
                'beta' : b,
                'seed' : seed_vector[i],
                'time_limit': time_lim,
                '#_sol_methods': len(COMPARE_SOLUTIONS)
            }

            pref_list_length = m # Assume pref_list_length = number of schools (as they do)
            MyData = DataGenEE(n, m, a, b, pref_list_length, False, seed_vector[i])

            # Generate the assignment from DA with Single Tie-Breaking with n_match samples
            A = DA_STB(MyData, n_match, 'GS', False, seed_vector[i], print_intermediate)

            # Find Stable improvement cycles à la Erdil and Ergin (2008)
            A_SIC = SIC_all_matchings(MyData, A, print_intermediate)

            row_data['avg_rank_DA'] = A.statistics()
            row_data['avg_rank_EE'] = A_SIC.statistics()
            row_data['n_assigned_DA'] = A.compute_n_assigned_students()
            row_data['n_assigned_EE'] = A_SIC.compute_n_assigned_students()

            # Run EADA if necessary:
            if ("SD_UPON_EADA" in COMPARE_SOLUTIONS) or ("SD_UPON_EADA_SAMPLE" in COMPARE_SOLUTIONS):
                A_EADAM = EADAM_STB(MyData, n_match, seed_vector[i], print_intermediate)
                row_data['avg_rank_EADA'] = A_EADAM.statistics()
                row_data['n_assigned_EADA'] = A_EADAM.compute_n_assigned_students()
            else:
                A_EADAM = A # Dummy assignment to avoid errors later on

            counter = 1
            for s in COMPARE_SOLUTIONS:
                if s == "SD_UPON_DA":
                    # Solve the formulations
                    # Note that we will use the matchings in 'A_SIC' in the first step of the master problem
                    # And we will sd-dominate the assignment 'A.assignment' (found by DA)
                    MyModel = ModelColumnGen(MyData, A_SIC, A.assignment, bool_identical_students, False, print_intermediate)
                    S = MyModel.Solve("CUTOFF", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, n_sol_pricingMinRank=n_sol_pricingMinRank, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, bool_supercolumn=bool_supercolumn, print_out=print_intermediate)
                    
                    row_data = add_line(row_data=row_data, S=S, s=s, sol_list=COMPARE_SOLUTIONS, counter=counter, A=A, A_SIC=A_SIC, A_EADAM=A_EADAM, bool_ColumnGen=bool_ColumnGen)
                elif s == "SD_UPON_EE":
                    MyModel = ModelColumnGen(MyData, A_SIC, A_SIC.assignment, bool_identical_students, False, print_intermediate)
                    S = MyModel.Solve("CUTOFF", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, n_sol_pricingMinRank=n_sol_pricingMinRank, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, bool_supercolumn=bool_supercolumn, print_out=print_intermediate)

                    row_data = add_line(row_data=row_data, S=S, s=s, sol_list=COMPARE_SOLUTIONS, counter=counter, A=A, A_SIC=A_SIC, A_EADAM=A_EADAM, bool_ColumnGen=bool_ColumnGen)
 
                elif s == "SD_UPON_EADA":
                    MyModel = ModelColumnGen(MyData, A_SIC, A_EADAM.assignment, bool_identical_students, False,  print_intermediate)
                    S = MyModel.Solve("CUTOFF", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, n_sol_pricingMinRank=n_sol_pricingMinRank, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, bool_supercolumn=bool_supercolumn, print_out=print_intermediate)
            
                    row_data = add_line(row_data=row_data, S=S, s=s, sol_list=COMPARE_SOLUTIONS, counter=counter, A=A, A_SIC=A_SIC, A_EADAM=A_EADAM, bool_ColumnGen=bool_ColumnGen)

                elif s == "SD_UPON_EE":
                    MyModel = ModelColumnGen(MyData, A_SIC, A_SIC.assignment, bool_identical_students, False, print_intermediate)
                    
                    MyModel.add_n_new_matchings(n_sol_extra_sample, seed_vector[i] + 100, print_out) # +100 to have different seed than initial sampling
                    
                    S = MyModel.Solve("CUTOFF", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, n_sol_pricingMinRank=n_sol_pricingMinRank, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, bool_supercolumn=bool_supercolumn, print_out=print_intermediate)

                    row_data = add_line(row_data=row_data, S=S, s=s, sol_list=COMPARE_SOLUTIONS, counter=counter, A=A, A_SIC=A_SIC, A_EADAM=A_EADAM, bool_ColumnGen=bool_ColumnGen)
 
                
                elif s == "SD_UPON_DA_SAMPLE":
                    MyModel = ModelColumnGen(MyData, A_SIC, A.assignment, bool_identical_students, False,  print_intermediate)
                    
                    MyModel.add_n_new_matchings(n_sol_extra_sample, seed_vector[i] + 100, print_out)
                    
                    S = MyModel.Solve("CUTOFF", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, n_sol_pricingMinRank=n_sol_pricingMinRank, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, bool_supercolumn=bool_supercolumn, print_out=print_intermediate)
                    
                    row_data = add_line(row_data=row_data, S=S, s=s, sol_list=sol_list, counter=counter, A=A, A_SIC=A_SIC, A_EADAM=A_EADAM, bool_ColumnGen=bool_ColumnGen)


                elif s == "SD_UPON_EADA_SAMPLE":
                    MyModel = ModelColumnGen(MyData, A_SIC, A_EADAM.assignment, bool_identical_students, False,  print_intermediate)

                    MyModel.add_n_new_matchings(n_sol_extra_sample, seed_vector[i] + 100, print_out)

                    S = MyModel.Solve("CUTOFF", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, n_sol_pricingMinRank=n_sol_pricingMinRank, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, bool_supercolumn=bool_supercolumn, print_out=print_intermediate)
            
                    row_data = add_line(row_data=row_data, S=S, s=s, sol_list=sol_list, counter=counter, A=A, A_SIC=A_SIC, A_EADAM=A_EADAM, bool_ColumnGen=bool_ColumnGen)
              
                # Write master objectives to csv file
                row_data_obj = [n,m,a,b,seed_vector[i], s, S.iter]
                for k in S.obj_master:
                    row_data_obj.append(k)

                with open(output_file_obj, 'a', newline='') as csvfile:
                    # Create a CSV writer object
                    writer = csv.writer(csvfile)
                    writer.writerow(row_data_obj)
                counter = counter + 1

            # Add comparisons in improving students between EE, DA, and EADA
            comp_EE_DA = A_SIC.compare(A.assignment)
            row_data['n_stud_impr_EE_DA'] = comp_EE_DA["n_students_improving"]    
            row_data['avg_rank_impr_EE_DA'] = comp_EE_DA["average_rank_increase"]  
            row_data['median_rank_impr_EE_DA'] = comp_EE_DA["median_rank_improvement"]  

            if ("SD_UPON_EADA" in COMPARE_SOLUTIONS) or ("SD_UPON_EADA_SAMPLE" in COMPARE_SOLUTIONS):
                comp_EADA_DA = A_EADAM.compare(A.assignment)
                comp_EADA_EE = A_EADAM.compare(A_SIC.assignment)
                comp_EE_EADA = A_SIC.compare(A_EADAM.assignment)

                row_data['n_stud_impr_EADA_DA'] = comp_EADA_DA["n_students_improving"]     
                row_data['avg_rank_impr_EADA_DA'] = comp_EADA_DA["average_rank_increase"]  
                row_data['median_rank_impr_EADA_DA'] = comp_EADA_DA["median_rank_improvement"]  

                row_data['n_stud_impr_EADA_EE'] = comp_EADA_EE["n_students_improving"]     
                row_data['avg_rank_impr_EADA_EE'] = comp_EADA_EE["average_rank_increase"]  
                row_data['median_rank_impr_EADA_EE'] = comp_EADA_EE["median_rank_improvement"] 

                row_data['n_stud_impr_EE_EADA'] = comp_EE_EADA["n_students_improving"]     
                row_data['avg_rank_impr_EE_EADA'] = comp_EE_EADA["average_rank_increase"]  
                row_data['median_rank_impr_EE_EADA'] = comp_EE_EADA["median_rank_improvement"]  



            # EXPORT TO CSV FILE
            with open(output_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writerow(row_data)


            

    return S_vector

def add_line(row_data: dict, S: SolutionReport, s, sol_list: list, counter: int, A: Assignment, A_SIC: Assignment, A_EADAM: Assignment, bool_ColumnGen: bool):
    # Store solution
    row_data[f'sol_{counter}_label'] = s
    row_data[f'{counter}_avg_rank_result'] = S.avg_ranks['result']
    row_data[f'{counter}_avg_rank_heur'] = S.avg_ranks['first_iter']
    row_data[f'{counter}_bool_CG'] = bool_ColumnGen
    row_data[f'{counter}_time'] = S.time
    row_data[f'{counter}_time_lim_exceeded'] = S.time_limit_exceeded
    row_data[f'{counter}_optimal'] = S.optimal  
    row_data[f'{counter}_n_iter'] = S.iter      
    row_data[f'{counter}_n_match'] = S.n_match
    row_data[f'{counter}_n_match_support'] = S.n_match_support
    row_data[f'{counter}_n_assigned'] = S.n_students_assigned

    comp_DA = S.A.compare(A.assignment)
    comp_EE = S.A.compare(A_SIC.assignment)
    if ("SD_UPON_EADA" in sol_list) or ("SD_UPON_EADA_SAMPLE" in sol_list):
        comp_EADA = S.A.compare(A_EADAM.assignment)

    row_data[f'{counter}_n_stud_impr_DA'] = comp_DA["n_students_improving"]     
    row_data[f'{counter}_avg_rank_impr_DA'] = comp_DA["average_rank_increase"]  
    row_data[f'{counter}_median_rank_impr_DA'] = comp_DA["median_rank_improvement"]   

    row_data[f'{counter}_n_stud_impr_EE'] = comp_EE["n_students_improving"]     
    row_data[f'{counter}_avg_rank_impr_EE'] = comp_EE["average_rank_increase"]  
    row_data[f'{counter}_median_rank_impr_EE'] = comp_EE["median_rank_improvement"]    
    if ("SD_UPON_EADA" in sol_list) or ("SD_UPON_EADA_SAMPLE" in sol_list):
        row_data[f'{counter}_n_stud_impr_EADA'] = comp_EADA["n_students_improving"]     
        row_data[f'{counter}_avg_rank_impr_EADA'] = comp_EADA["average_rank_increase"]  
        row_data[f'{counter}_median_rank_impr_EADA'] = comp_EADA["median_rank_improvement"]   

    return row_data





def SimulationEE(COMPARE_SOLUTIONS: list, n_students_schools: list, alpha: list, beta: list, n_iterations_simul: int, n_match: int, n_sol_extra_sample: int, time_lim: int, seed: int, n_sol_pricing: int, n_sol_pricingMinRank:int, gap_pricing: float, MIPGap: float, bool_ColumnGen: bool, bool_supercolumn: bool, bool_identical_students: bool, print_out = False):
    """
    Evaluate effect of bug in code EE

    - COMPARE_SOLUTIONS: a list of strings that contains the solutions we want to compare
    - n_students_schools, alpha, beta: lists containing the corresponding parameter values
    - n_students_schools is a list of pairs of values to evaluate: e,g,. [[100, 2], [200, 4], [400, 8]]
    - n_sol_extra_sample: number of additional weakly stable matchings sampled when evaluating nethods with larger initial sample sizes

    For column generation
    - n_match: number of solutions returned by pricing problem
    - n_sol_pricingMinRank: number of solutions returned by the pricing problem that minimizes the average rank
    - gap_pricing: optimality gap used for the solutions included in the solution pool in the pricing problem
    - MIPGap: gap for pricing problem
    - bool_ColumnGen: if True: perform entire column generation for time_limit period
                      if False: only perform first iteration, and don't build pricing problem
    - bool_supercolumn: boolean to determine whether you want to add an artificial column to the model when the model is infeasible?
    - bool_identical_students (bool): if True, give identical students the same probabilities

    For each combination of parameter values (n_stud, n_school, alpha, beta), we start seed from same value
    This makes it easier to reproduce the results later on
    """
    print_intermediate = print_out
    print_counter = True

    # Create directory if it doesn't exist
    os.makedirs('Simulation Results', exist_ok=True)

    S_vector = []

    now = time.strftime('%Y-%m-%d_%H%M%S')

    # The different possible solutions
    sol_list = ["SD_UPON_DA", "SD_UPON_EE", "SD_UPON_EADA", "SD_UPON_DA_SAMPLE", "SD_UPON_EE_SAMPLE", "SD_UPON_EADA_SAMPLE"]
    for sol in COMPARE_SOLUTIONS:
        if sol not in sol_list:
            raise ValueError(f"Invalid value: '{sol}'. Allowed values are: {sol_list}")   
        
    # Create csv file
    output_file = 'Simulation Results/SIM_' + now + '_EE_bug.csv'
    headers = ['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'time_limit', '#_sol_methods', 'avg_rank_DA', 'avg_rank_EE', 'avg_rank_EE_bug', 'diff_assignment', 'n_assigned_DA', 'n_assigned_EE']

    
    # Add comparisons between the number of students who improve in the base cases (e.g., how many students improve in EE compared to DA)
    label_name = f'n_stud_impr_EE_DA'
    headers.append(label_name)
    label_name = f'avg_rank_impr_EE_DA'
    headers.append(label_name)
    label_name = f'median_rank_impr_EE_DA'
    headers.append(label_name)

    label_name = f'n_stud_impr_EE_bug_DA'
    headers.append(label_name)
    label_name = f'avg_rank_impr_EE_bug_DA'
    headers.append(label_name)
    label_name = f'median_rank_impr_EE_bug_DA'
    headers.append(label_name)


    # Create csv in which final results will be stored
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader() # Writes the header row from 'headers

    

    # Create pandas dataframe to store results along the way
    df = pd.DataFrame(columns=headers)


    total_combinations = len(n_students_schools) * len(alpha) * len(beta)

    for k, b, a in tqdm(itertools.product(n_students_schools, beta, alpha), total = total_combinations, desc='Data instances', unit = 'inst', disable= not print_counter):
        # Generate data using data generation by Erdil & Ergin (2008)
        random.seed(seed)

        n = k[0] # Number of students
        m = k[1] # Number of schools

        seed_vector = []
        for i in range(n_iterations_simul):
            seed_vector.append(random.randint(0,1000000000))

        for i in tqdm(range(n_iterations_simul), desc = 'iterations', total = n_iterations_simul, unit = 'iter', disable = not print_out):
            if print_out:
                print('\nn,m,alpha, beta, seed', n, m, a, b, seed_vector[i])
            
            row_data = {
                'n_stud' : n,
                'n_schools' : m,
                'alpha' : a,
                'beta' : b,
                'seed' : seed_vector[i],
                'time_limit': time_lim,
                '#_sol_methods': len(COMPARE_SOLUTIONS)
            }

            pref_list_length = m # Assume pref_list_length = number of schools (as they do)
            MyData = DataGenEE(n, m, a, b, pref_list_length, False, seed_vector[i])

            # Generate the assignment from DA with Single Tie-Breaking with n_match samples
            A = DA_STB(MyData, n_match, 'GS', False, seed_vector[i], print_intermediate)

            # Find Stable improvement cycles à la Erdil and Ergin (2008)
            A_SIC = SIC_all_matchings(MyData, A, print_intermediate)
            A_SIC_bug = SIC_all_matchings_with_mistake_EE(MyData, A, False)

            row_data['avg_rank_DA'] = A.statistics()
            row_data['avg_rank_EE'] = A_SIC.statistics()
            row_data['avg_rank_EE_bug'] = A_SIC_bug.statistics()
            row_data['diff_assignment'] = np.array_equal(A_SIC.assignment, A_SIC_bug.assignment)

            # Add comparisons in improving students between EE, DA, and EADA
            comp_EE_DA = A_SIC.compare(A.assignment)
            row_data['n_stud_impr_EE_DA'] = comp_EE_DA["n_students_improving"]    
            row_data['avg_rank_impr_EE_DA'] = comp_EE_DA["average_rank_increase"]  
            row_data['median_rank_impr_EE_DA'] = comp_EE_DA["median_rank_improvement"]  

            comp_EE_bug_DA = A_SIC_bug.compare(A.assignment)
            row_data['n_stud_impr_EE_bug_DA'] = comp_EE_bug_DA["n_students_improving"]    
            row_data['avg_rank_impr_EE_bug_DA'] = comp_EE_bug_DA["average_rank_increase"]  
            row_data['median_rank_impr_EE_bug_DA'] = comp_EE_bug_DA["median_rank_improvement"]  

           

            # EXPORT TO CSV FILE
            with open(output_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writerow(row_data)      

    return S_vector

