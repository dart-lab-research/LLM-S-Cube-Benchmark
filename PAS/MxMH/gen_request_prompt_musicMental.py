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


def generate_condq_prompt(value, mappings, row):
    # 初始化提示信息
    cond_info_prompt = "All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information, please answer the following questions based on the given conditions:"

    # 从 row 中提取数据并格式化为提示内容
    background_data = f"This individual, identified by id: {int(row['index'])}, is a {int(row['Age'])}-year-old, with a favorite music genre of {mappings['Fav genre'].get(str(value['Fav genre']))}. The individual listens to music for {row['Hours per day']} each day and {mappings['While working'].get(str(value['While working']))}." \
                      f"Additionally, the individual {mappings['Instrumentalist'].get(str(value['Instrumentalist']))} an instrument regularly. {mappings['Composer'].get(str(value['Composer']))}, and {mappings['Exploratory'].get(str(value['Exploratory']))}"
    # 工作习惯和环境描述
    partial_info = f"""
    The individual’s music listening habits include the following frequencies for various genres:
    - Classical music: {mappings['Frequency [Classical]'].get(str(value['Frequency [Classical]']))},
    - Country music: {mappings['Frequency [Country]'].get(str(value['Frequency [Country]']))},
    - EDM: {mappings['Frequency [EDM]'].get(str(value['Frequency [EDM]']))},
    - Folk: {mappings['Frequency [Folk]'].get(str(value['Frequency [Folk]']))},
    - Gospel: {mappings['Frequency [Gospel]'].get(str(value['Frequency [Gospel]']))},
    - Hip hop: {mappings['Frequency [Hip hop]'].get(str(value['Frequency [Hip hop]']))},
    - Jazz: {mappings['Frequency [Jazz]'].get(str(value['Frequency [Jazz]']))},
    - K pop: {mappings['Frequency [K pop]'].get(str(value['Frequency [K pop]']))},
    - Latin: {mappings['Frequency [Latin]'].get(str(value['Frequency [Latin]']))},
    - Lofi: {mappings['Frequency [Lofi]'].get(str(value['Frequency [Lofi]']))},
    - Metal: {mappings['Frequency [Metal]'].get(str(value['Frequency [Metal]']))},
    - Pop: {mappings['Frequency [Pop]'].get(str(value['Frequency [Pop]']))},
    - R&B: {mappings['Frequency [R&B]'].get(str(value['Frequency [R&B]']))},
    - Rap: {mappings['Frequency [Rap]'].get(str(value['Frequency [Rap]']))},
    - Rock: {mappings['Frequency [Rock]'].get(str(value['Frequency [Rock]']))},
    - Video game music: {mappings['Frequency [Video game music]'].get(str(value['Frequency [Video game music]']))}.
    """

    # 将工作信息和国家信息合并
    cond_info_prompt += background_data
    cond_info_prompt += partial_info

    # 假设 gen_prompt_from_fields 是你用来生成基于其他字段的额外提示信息的函数

    question_prompt = """
    Now that you are role-playing this person based on the provided demographic and music-related information, 
    predict the individual’s levels of Anxiety, Depression, Insomnia, and OCD. 
    Please provide the predicted values for each condition as a single numeric value on a scale from 0 to 10, where 0 indicates none and 10 indicates extreme. 
    Do not include any additional text or explanation.
    
    Anxiety:
    Depression:
    Insomnia:
    OCD:


    """

    few_shot_prompt = ("""Here are some example,
Example One: 
    - This individual is a 18-year-old, with a favorite music genre of Latin. 
    - The individual listens to music for 3 hours each day and listens to music while working. 
    - Additionally, the individual plays an instrument regularly. 
    - The respondent composes music, and who actively explores new artists or genres.
 The individual’s music listening habits include the following frequencies for various genres:
    - Classical music: The respondent listens to classical music rarely,
    - Country music: The respondent never listens to country music,
    - EDM: The respondent listens to EDM rarely,
    - Folk: The respondent never listens to folk music,
    - Gospel: The respondent never listens to gospel music,
    - Hip hop: The respondent sometimes listens to hip hop,
    - Jazz: The respondent never listens to jazz music,
    - K pop: The respondent listens to K pop very frequently,
    - Latin: The respondent listens to Latin music very frequently,
    - Lofi: The respondent listens to Lofi music rarely,
    - Metal: The respondent never listens to metal music,
    - Pop: The respondent listens to pop music very frequently,
    - R&B: The respondent sometimes listens to R&B,
    - Rap: The respondent listens to rap very frequently,
    - Rock: The respondent never listens to rock music,
    - Video game music: The respondent sometimes listens to video game music.

The answers of this respondent are 3,0,1,0

Example Two: 
    - This individual is a 63-year-old, with a favorite music genre of Rock. 
    - The individual listens to music for 1.5 hours each day and listens to music while working. 
    - Additionally, the individual dose not play an instrument regularly. 
    - The respondent does not compose music, and who actively explores new artists or genres.
 The individual’s music listening habits include the following frequencies for various genres:
   - Classical music: The respondent sometimes listens to classical music,
    - Country music: The respondent never listens to country music,
    - EDM: The respondent never listens to EDM,
    - Folk: The respondent listens to folk music rarely,
    - Gospel: The respondent sometimes listens to gospel music,
    - Hip hop: The respondent listens to hip hop rarely,
    - Jazz: The respondent listens to jazz very frequently,
    - K pop: The respondent listens to K pop rarely,
    - Latin: The respondent sometimes listens to Latin music,
    - Lofi: The respondent listens to Lofi music rarely,
    - Metal: The respondent never listens to metal music,
    - Pop: The respondent sometimes listens to pop music,
    - R&B: The respondent sometimes listens to R&B,
    - Rap: The respondent listens to rap rarely,
    - Rock: The respondent listens to rock very frequently,
    - Video game music: The respondent listens to video game music rarely.

The answers of this respondent are 7,2,2,1

Example Three: 
    - This individual is a 18-year-old, with a favorite music genre of Video game music. 
    - The individual listens to music for 4 hours each day and does not listen to music while working. 
    - Additionally, the individual dose not play an instrument regularly. 
    - The respondent does not compose music, and who does not actively explore new artists or genres
 The individual’s music listening habits include the following frequencies for various genres:
   - Classical music: The respondent never listens to classical music,
    - Country music: The respondent never listens to country music,
    - EDM: The respondent listens to EDM very frequently,
    - Folk: The respondent never listens to folk music,
    - Gospel: The respondent never listens to gospel music,
    - Hip hop: The respondent listens to hip hop rarely,
    - Jazz: The respondent listens to jazz rarely,
    - K pop: The respondent listens to K pop very frequently,
    - Latin: The respondent never listens to Latin music,
    - Lofi: The respondent sometimes listens to Lofi music,
    - Metal: The respondent sometimes listens to metal music,
    - Pop: The respondent listens to pop music rarely,
    - R&B: The respondent never listens to R&B,
    - Rap: The respondent listens to rap rarely,
    - Rock: The respondent listens to rock very frequently,
    - Video game music: The respondent listens to video game music rarely.

The answers of this respondent are 7,7,10,2

Example Four: 
    - This individual is a 21-year-old, with a favorite music genre of K pop. 
    - The individual listens to music for 1 hours each day and listens to music while working. 
    - Additionally, the individual dose not play an instrument regularly. 
    - The respondent does not compose music, and who actively explore new artists or genres.
 The individual’s music listening habits include the following frequencies for various genres:
   - Classical music: The respondent never listens to classical music,
    - Country music: The respondent never listens to country music,
    - EDM: The respondent listens to EDM rarely,
    - Folk: The respondent never listens to folk music,
    - Gospel: The respondent never listens to gospel music,
    - Hip hop: The respondent listens to hip hop very frequently,
    - Jazz: The respondent listens to jazz rarely,
    - K pop: The respondent listens to K pop very frequently,
    - Latin: The respondent never listens to Latin music,
    - Lofi: The respondent sometimes listens to Lofi music,
    - Metal: The respondent never listens to metal music,
    - Pop: The respondent sometimes listens to pop music,
    - R&B: The respondent sometimes listens to R&B,
    - Rap: The respondent listens to rap rarely,
    - Rock: The respondent never listens to rock music,
    - Video game music: The respondent listens to video game music rarely.

The answers of this respondent are 5,3,5,3

Example Five:
    - This individual is a 16-year-old, with a favorite music genre of Hip hop. 
    - The individual listens to music for 12 hours each day and listens to music while working. 
    - Additionally, the individual dose not play an instrument regularly. 
    - The respondent composes music, and who actively explore new artists or genres.
 The individual’s music listening habits include the following frequencies for various genres:
    - Classical music: The respondent listens to classical music rarely,
    - Country music: The respondent never listens to country music,
    - EDM: The respondent sometimes listens to EDM,
    - Folk: The respondent listens to folk music rarely,
    - Gospel: The respondent never listens to gospel music,
    - Hip hop: The respondent listens to hip hop very frequently,
    - Jazz: The respondent listens to jazz rarely,
    - K pop: The respondent never listens to K pop,
    - Latin: The respondent never listens to Latin music,
    - Lofi: The respondent never listens to Lofi music,
    - Metal: The respondent sometimes listens to metal music,
    - Pop: The respondent sometimes listens to pop music,
    - R&B: The respondent listens to R&B rarely,
    - Rap: The respondent sometimes listens to rap,
    - Rock: The respondent listens to rock very frequently,
    - Video game music: The respondent never listens to video game music.

The answers of this respondent are 5,7,10,0

""")

#    question_prompt += few_shot_prompt
    #     prompt_format = """Your response should consist of just Five letter separated by four commas without any additional text or explanation.Example output:
    # C,D,B,E,D"Now, based on the employee's details, predict their ratings for the specified workplace factors.
    # """
    prompt_format = """For the prediction task, the output should consist of four numeric values, each corresponding to the predicted levels of Anxiety, Depression, Insomnia, and OCD. 
    These values should be given as a single line, separated by spaces. Each value should be between 0 and 10, where 0 indicates none and 10 indicates extreme. Your response should consist of just Five letter separated by four commas without any additional text or explanation.
    "Your response should consist of just four numbers separated by three commas without any additional text or explanation.
    """
    return cond_info_prompt + question_prompt + prompt_format  # + "\n"
