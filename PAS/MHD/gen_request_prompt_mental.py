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


def generate_condq_prompt(value, mappings, cond_fields,timestamp, option_field, number_field):
    cond_info_prompt = f"All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information, please answer the following questions based on the given conditions:"
    country_data = f"The timestamp respondent interviewed at {timestamp},"
    cond_info_prompt += country_data
    cond_info_prompt += gen_prompt_from_fields(value, mappings, cond_fields, option_field, number_field)
    question_prompt = """ Now that you are role-playing this person based on the above information. Under these conditions, . The selections are follows: 
    Quesion oneGrowing Stress. The current level of stress the respondent feels:
    For Growing Stress selection From A to C:
    "B": "Maybe.",
    "A": "Yes.",
    "C": "No.",

    Question two Mood Swings.Does the respondent experience sudden mood swings?
    For Mood Swings selection From A to C:
    "B": "Medium.",
    "C": "Low",
    "A": "High.",

    Question three Coping Struggles. Does the respondent have difficulty coping with pressure or stress?
    For Coping Struggles selection From A or B:
    "A": "Yes.",
    "B": "No.",

    Qestion four Social Weakness. Does the respondent find it difficult to interact socially or maintain relationships?
    For Social weakness selection From A to C:
    "A": "Yes.",
    "B": "Maybe.",
    "C": "No."\n
    """
    few_shot_prompt = ("""Here are some examples for you:\n 
    Example one: The timestamp respondent interviewed at 2014-08-27 11:29:31, respondent experienced does not have any changes in sleeping habits or patterns, the respondent has a known history of mental health disorders, the respondent is not willing to participate in a mental health interview, the respondent is uncertain about available mental health care options or whether they use them, the respondent does not show much interest or motivation in their work. The answer of four question is A,B,B,A.\n
    Example two: The timestamp respondent interviewed at 2014-08-28 22:46:40, respondent experienced does not have any changes in sleeping habits or patterns the respondent has a known history of mental health disorders, the respondent is willing to participate in a mental health interview, the respondent is uncertain about available mental health care options or whether they use them, the respondent does not show much interest or motivation in their work. The answer of four questions is C, C, A, A.\n
    Example three: The timestamp respondent interviewed at 2014-08-28 11:10:53, respondent experienced does not have any changes in sleeping habits or patterns the respondent does not have a known history of mental health disorders, the respondent is unwilling to participate in a mental health interview, the respondent is unsure about available mental health care options or whether they use them, the respondent shows some interest or motivation in their work. The answer of four questions is B, C, A, A.\n
    Example four: The timestamp respondent interviewed at 2014-08-27 11:37:59, respondent experienced does not have any changes in sleeping habits or patterns, the respondent has a known history of mental health disorders, the respondent is not willing to participate in a mental health interview, the respondent is uncertain about available mental health care options or whether they use them, the respondent does not show much interest or motivation in their work. The answer of four question is A,B,B,A.\n
    Example five: The timestamp respondent interviewed at 2014-09-01 09:12:15, respondent experienced does not have any changes in sleeping habits or patterns the respondent does not have a known history of mental health disorders, the respondent is willing to participate in a mental health interview, the respondent is uncertain about available mental health care options or whether they use them, the respondent shows some interest or motivation in their work. The answer of four questions is A, B, A, B.\n
    
""")

    #question_prompt += few_shot_prompt
    prompt_format = "Your response should consist of just four letter separated by Three commas without any additional text or explanation. First answer from A-C,second answer from A-C,thrid answer A or B, fourth answer from A-C"
    return cond_info_prompt + question_prompt + prompt_format  # + "\n"



