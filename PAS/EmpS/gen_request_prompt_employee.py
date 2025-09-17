# 生成提示词，根据回答者的基础信息，进行角色扮演，并给出预测
import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from utils import load_json


# 定义一个函数，根据条件转换值
def convert_value(val):
    try:
        float_val = float(val)
        # 如果浮点数是整数，则转换为整数
        if float_val.is_integer():
            return int(float_val)
        else:
            return float_val
    except ValueError:
        return val


def gen_prompt_from_fields(value, mappings, info_fields, option_field, number_field):
    # 为每一个特征field生成一句话info_desc
    info_prompt = ""

    for field in info_fields:
        info_desc = ""
        # 检查字段是否存在于 value 中
        if field in value:
            fieldvalue = str(value[field])
        else:
            # print(f"Warning: Field '{field}' not found in value. Using default value.")
            fieldvalue = "N/A"  # 或者你可以选择其他的默认值

        # 选择题
        if field in option_field:
            info_desc = mappings[field].get(fieldvalue, "Unknown")  # 如果 mappings 中没有对应值，返回 "Unknown"
        # 数值题
        elif field in number_field:
            info_desc = f"{mappings.get(field, '')} {fieldvalue}."
        else:
            print("ERROR 1")
            print(field, fieldvalue)

        info_prompt += info_desc + " "

    return info_prompt


# Function to map values
def map_value(mappings, category, value):
    if category in mappings and value in mappings[category]:
        # print(mappings[category][value])
        return mappings[category][value]
    return str(value)


def generate_condq_prompt(value, mappings,  row):
    # 初始化提示信息
    cond_info_prompt = "All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information, please answer the following questions based on the given conditions:"

    # 从 row 中提取数据并格式化为提示内容
    background_data = f"This person, identified by EmpID: {int(row['EmpID'])}, is a {int(row['Age'])}-year-old {row['Gender']} who is {mappings['MaritalStatus'].get(str(value['MaritalStatus']))} and {mappings['EduLevel'].get(str(value['EduLevel']))}"

    # 工作习惯和环境描述
    work_info = f"""
    When describing this employee’s work habits and environment:
    - {mappings['JobLevel'].get(str(value['JobLevel']))} in the IT department with {int(row['Experience'])} years of experience and {mappings['EmpType'].get(str(value['EmpType']))}.
    - This person engages in {float(row['PhysicalActivityHours'])} hours of physical activity per week and sleeps an average of {float(row['SleepHours'])} hours per night.
    - This person commutes {float(row['CommuteDistance'])} km to work through {mappings['CommuteMode'].get(str(value['CommuteMode']))} and has worked for {int(row['NumCompanies'])} different companies throughout their career.
    - In the current role, this person is part of a team of {float(row['TeamSize'])} people and {mappings['haveOT'].get(str(value['haveOT']))}. This person also completes {float(row['TrainingHoursPerYear'])} hours of training per year.
    """

    # 将工作信息和国家信息合并
    cond_info_prompt += background_data
    cond_info_prompt += work_info

    # 假设 gen_prompt_from_fields 是你用来生成基于其他字段的额外提示信息的函数

    question_prompt ="""
    Now that you are role-playing this person based on the above information. Under these conditions, Now based on the known information, predict the following information question for this respondent:
    
    Question one Work-Life Balance (WLB): This person’s balance between work responsibilities and personal life:
    "A": "Respondent has a very poor work-life balance, with work responsibilities significantly overwhelming personal life.",
    "B": "Respondent has a poor work-life balance, with work responsibilities often affecting personal life.",
    "C": "Respondent feels neutral about work-life balance, with work and personal life being reasonably manageable.",
    "D": "Respondent has a good work-life balance, with personal life and work responsibilities in harmony.",
    "E": "Respondent has an excellent work-life balance, with personal life significantly more important than work responsibilities."

    
    Question two Work Environment (WorkEnv): The overall quality of this person’s work setting.
    "A": "Respondent works in a very poor environment, with many distractions and discomforts that hinder productivity.",
    "B": "Respondent works in a subpar environment, where the conditions are less than ideal and sometimes cause frustration.",
    "C": "Respondent works in an average environment, with reasonable comfort and productivity but no outstanding features.",
    "D": "Respondent works in a good environment, with comfortable conditions that promote productivity and well-being.",
    "E": "Respondent works in an excellent environment, with top-tier comfort and all factors working in favor of productivity."

    
    Question three The amount of work this person handles regularly.
    "A": "Respondent has an extremely light workload, rarely being assigned significant tasks.",
    "B": "Respondent has a light workload, with tasks being manageable but sometimes lacking challenge.",
    "C": "Respondent has a moderate workload, with tasks that require consistent effort and attention.",
    "D": "Respondent has a heavy workload, with numerous tasks requiring significant effort and time.",
    "E": "Respondent has an extremely heavy workload, often being overloaded with tasks and responsibilities."

    Question four The degree of stress this person experiences at work.
    "A": "Respondent experiences very little to no stress at work, with most tasks being manageable and calm.",
    "B": "Respondent experiences low stress at work, but occasionally there are tasks that may cause some tension.",
    "C": "Respondent experiences moderate stress at work, balancing between challenging tasks and manageable workload.",
    "D": "Respondent experiences high stress at work, with tight deadlines and difficult tasks being frequent.",
    "E": "Respondent experiences extremely high stress at work, with constant pressure and overwhelming tasks."

    Question five This person’s overall satisfaction with their job.
    "A": "Respondent is very dissatisfied with his or her job.",
    "B": "Respondent is dissatisfied with his or her job.",
    "C": "Respondent feels neutral about his or her job.",
    "D": "Respondent is satisfied with his or her job.",
    "E": "Respondent is very satisfied with his or her job."

    
    
    
    """


    few_shot_prompt = ("""Here are some example,
    Example one: 
- The respondent is at the Mid job level. in the IT department with 12 years of experience and is employed full-time..
    - This person engages in 1.8 hours of physical activity per week and sleeps an average of 7.9 hours per night.
    - This person commutes 15.0 km to work through The respondent commutes by car. and has worked for 4 different companies throughout their career.
    - In the current role, this person is part of a team of 11.0 people and None. This person also completes 36.0 hours of training per year.
The answers of this respondent are A,A,B,B,E


Example two:
- The respondent is at the Junior job level. in the IT department with 4 years of experience and is employed full-time.
    - This person engages in 1.8 hours of physical activity per week and sleeps an average of 5.8 hours per night.
    - This person commutes 29.0 km to work through The respondent commutes by car. and has worked for 1 different companies throughout their career.
    - In the current role, this person is part of a team of 8.0 people and None. This person also completes 22.0 hours of training per year.
The answers of this respondent are B,B,A,A,E

Example three:
- The respondent is at the Junior job level. in the IT department with 3 years of experience and is employed part-time..
    - This person engages in 3.5 hours of physical activity per week and sleeps an average of 6.5 hours per night.
    - This person commutes 7.0 km to work through The respondent commutes by bike. and has worked for 1 different companies throughout their career.
    - In the current role, this person is part of a team of 6.0 people and None. This person also completes 21.5 hours of training per year.
The answers of this respondent are E,C,B,A,C

Example four:
- The respondent is at the Junior job level. in the IT department with 2 years of experience and is employed part-time..
    - This person engages in 2.0 hours of physical activity per week and sleeps an average of 7.7 hours per night.
    - This person commutes 25.0 km to work through The respondent commutes by car. and has worked for 0 different companies throughout their career.
    - In the current role, this person is part of a team of 7.0 people and None. This person also completes 21.0 hours of training per year.
The answers of this respondent are C,B,B,A,C

Example five:
- The respondent is at the Intern/Fresher job level. in the IT department with 0 years of experience and is employed part-time..
    - This person engages in 0.6 hours of physical activity per week and sleeps an average of 6.2 hours per night.
    - This person commutes 11.0 km to work through The respondent commutes by public transport. and has worked for 0 different companies throughout their career.
    - In the current role, this person is part of a team of 30.0 people and None. This person also completes 10.0 hours of training per year.
The answers of this respondent are D,B,C,A,D
""")
    #question_prompt += few_shot_prompt
#     prompt_format = """Your response should consist of just Five letter separated by four commas without any additional text or explanation.Example output:
# C,D,B,E,D"Now, based on the employee's details, predict their ratings for the specified workplace factors.
# """
    prompt_format = """Your response should consist of just Five letter separated by four commas without any additional text or explanation.
    """
    return cond_info_prompt +  question_prompt+prompt_format  # + "\n"
