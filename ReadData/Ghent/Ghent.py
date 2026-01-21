import pandas as pd
from src.Data import *
import csv

# Coded with help of ChatGPT


def read_dataGhent2024(print_out = False):
    file_path = "ReadData/Ghent/2024/2024_pref.xlsx"
    sheet_name = "Overzicht keuzes en details_202"
    student_col = "IdKind"
    school_col = "SchoolNummer"
    school_type_col = "LlGroep"
    rank_col = "Voorkeur"
    priority_col = "Voorrangsgroep"

    file_path_cap = "ReadData/Ghent/2024/2024_Cap.xlsx" # for capacities
    sheet_name_cap = "Vrije plaatsen_20240321111855"
    cap_col = "Capaciteit"
    school_col_cap = "Schoolnummer"
    school_type_col_cap = "Groep"

    # PREFERENCES
    df = pd.read_excel("ReadData/Ghent/2024/2024_pref.xlsx")

    # Read Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Drop rows with missing essential data (optional but recommended)
    df = df.dropna(subset=[student_col, school_col, rank_col])

    # Ensure correct sorting: first by student, then by preference rank
    df = df.sort_values(by=[student_col, rank_col])

    # Student IDs (order matters for pref!)
    df[student_col] = df[student_col].astype(str)
    ID_stud = sorted(df[student_col].unique().tolist())

    # School IDs (global list, order arbitrary but fixed)
    df["SchoolID"] = df.apply(
        lambda r: make_school_id(r[school_col], r[school_type_col]),
        axis=1
    )
    ID_school = sorted(df["SchoolID"].unique().tolist())

    # Build preference list
    pref = (
        df.groupby(student_col)["SchoolID"]
        .apply(list)
        .tolist()
    )

    # PRIORITIES
    # Explicit priority order
    priority_order = ["BZKP", "KP", "BZ", "REST"]

    prior = []

    for school in ID_school:
        df_school = df[df["SchoolID"] == school]

        school_prior = []

        # Add normal priority groups
        for group in priority_order:
            students_in_group = (
                df_school[df_school[priority_col] == group][student_col]
                .tolist()
            )

            if len(students_in_group) == 0:
                continue
            elif len(students_in_group) == 1:
                school_prior.append(students_in_group[0])
            else:
                school_prior.append(tuple(students_in_group))

        # Add students who did NOT rank this school
        students_in_prior = []
        for group in school_prior:
            if isinstance(group, tuple):
                students_in_prior.extend(group)
            else:
                students_in_prior.append(group)

        missing_students = [s for s in ID_stud if s not in students_in_prior]

        if len(missing_students) == 1:
            school_prior.append(missing_students[0])
        elif len(missing_students) > 1:
            school_prior.append(tuple(missing_students))

        prior.append(school_prior)

    
    # CAPACITIES
    df2 = pd.read_excel(file_path_cap, sheet_name=sheet_name_cap)
    df2 = df2.dropna(subset=[school_col_cap, cap_col])

    df2["SchoolID"] = df2.apply(
        lambda r: make_school_id_cap(r[school_col_cap], r[school_type_col_cap]),
        axis=1
    )

    # Build a dictionary {school_id: capacity}
    cap_dict = (
        df2.set_index("SchoolID")[cap_col]
        .to_dict()
    )

    # Align capacities with ID_school
    cap = []
    for school in ID_school:
        if school not in cap_dict:
            raise ValueError(f"Capacity missing for school {school}")
        cap.append(int(cap_dict[school]))


    return Data(n_stud=len(ID_stud), 
                n_schools=len(ID_school), 
                pref=pref, 
                prior=prior, 
                cap=cap, 
                ID_stud=ID_stud, 
                ID_school=ID_school, 
                file_name=file_path)


def make_school_id(schoolnummer, ligroep):
    """
    Create a unique school-stream ID.
    """
   
    if pd.isna(ligroep):
        raise ValueError(f"LlGroep missing for school {schoolnummer}")

    ligroep = ligroep.strip()  # defensive cleanup

    if ligroep == "1ste secundair stroom A":
        suffix = "_A"
    elif ligroep == "1ste secundair stroom B":
        suffix = "_B"
    elif ligroep == "1ste buitengewoon secundair stroom A":
        suffix = "_A_BUSO"
    elif ligroep == "1ste buitengewoon secundair stroom B":
        suffix = "_B_BUSO"
    else:
        raise ValueError(
            f"School type '{ligroep}' of school {schoolnummer} not defined."
        )

    return f"{schoolnummer}{suffix}"

def make_school_id_cap(schoolnummer, ligroep):
    """
    Create a unique school-stream ID.
    """
    if pd.isna(ligroep):
        raise ValueError(f"LlGroep missing for school {schoolnummer}")

    ligroep = ligroep.strip()  # defensive cleanup

    if ligroep == "A-stroom":
        suffix = "_A"
    elif ligroep == "B-stroom":
        suffix = "_B"
    elif ligroep == "1A BUSO":
        suffix = "_A_BUSO"
    elif ligroep == "1B BUSO":
        suffix = "_B_BUSO"
    else:
        raise ValueError(
            f"School type '{ligroep}' of school {schoolnummer} not defined."
        )

    return f"{schoolnummer}{suffix}"


