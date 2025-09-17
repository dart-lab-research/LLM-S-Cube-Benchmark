import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from prompt_gss import load_mapping, map_val2lab,map_sex
import re
from utils import apik
from openai import OpenAI
client = OpenAI(api_key=apik)
def ask_gpt3(client, messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        timeout = 60,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content

def ask_gpt4(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        timeout = 60,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content


few_examples = '''
This person is married, has an education level of 6 years of college, earns an income of unknown, is currently Retired, works as a unknown, and belongs to a household type of Married Couple, No Children.
The answer is 2.0

This person is never married, has an education level of 4 years of college, earns an income of $25,000 or more, is currently With a Job, But Not at Work Because of Temporary Illness, Vacation, Strike, works as a unknown, and belongs to a household type of Married Couple, No Children.
The answer is 1.0

This person is never married, has an education level of 12th grade, earns an income of $25,000 or more, is currently Working Full Time, works as a unknown, and belongs to a household type of Cohabitating Couple, No Children.
The answer is 1.0

This person is widowed, has an education level of 3 years of college, earns an income of $25,000 or more, is currently Retired, works as a unknown, and belongs to a household type of Single Adult.
The answer is 3.0
'''
def ol3(messages):

    stream = ollama.chat(
        model='llama3:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def ol31(messages):

    stream = ollama.chat(
        model='llama3.1:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def base_info(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/GSS/mappings/mapping.json')  
    # id = row['id']
    # age = row['age']
    # sibs = row['sibs']
    # race = map_val2lab('race', float(row['racecen1']), mapping)  
    # sex = map_sex(float(row['sex']), mapping)  

    #deal with empty values
    id = row.get('id', 'Unknown ID')  # Default to 'Unknown ID' if not present  
    age = row.get('age', '')  # Default to empty string if not present  
    sibs = row.get('sibs', '0')  # Default to '0' if not present  
    racecen1 = row.get('racecen1', '')  # Get racecen1 value  
    sex_value = row.get('sex', '')  # Get sex value  

    # Handling racecen1 conversion safely  
    try:  
        race = map_val2lab('race', float(racecen1), mapping) if racecen1 else 'Unknown race'  
    except ValueError:  
        race = 'Invalid race value'  

    # Handling sex conversion safely  
    try:  
        sex = map_sex(float(sex_value), mapping) if sex_value else 'Unknown sex'  
    except ValueError:  
        sex = 'Invalid sex value'  
    baseInfo_parts = []  
    if id:  
        baseInfo_parts.append(f"identified by ID: {id}")  
    if age:  
        baseInfo_parts.append(f"{age}-year-old")  
    if sex:  
        baseInfo_parts.append(sex)  
    if race:  
        baseInfo_parts.append(f"belonging to the {race} community, which is known for its cultural heritage")  
    # Join the parts, ensuring proper sentence structure  
    if baseInfo_parts:  
        baseInfo_description = "This person, " + ", ".join(baseInfo_parts) + ".\n"  
    else:  
        baseInfo_description = "This person has not provided sufficient information.\n"  
    # Constructing the description  
    description_parts = []  
    if sibs:  
        description_parts.append(f"they have {sibs} siblings, which suggests a lively household filled with shared experiences and memories.")  
    else:  
        description_parts.append("they have no siblings, suggesting a quieter household.")  

    description = "In their family, " + " ".join(description_parts) + " This person's background and family dynamics contribute to their unique perspective on life.\n\n"  
    #print(baseInfo_description+description)
    

    database_background = "The General Social Survey (GSS), conducted since 1972 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.\n"
    role_play_info = "You are a statistician and a social survey expert. I will give you some information about this person's response and attitude toward different policies in GSS2022 dataset, which requires you to accurately analyze this person's behavior and make predictions about other policy responses of this person.\n\n "
    
    return database_background, role_play_info, baseInfo_description, description

def hh_info(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/GSS/mappings/mapping.json')  
    #婚姻，教育，收入，工作状态，工作职务，家庭构成
    marital = map_val2lab('marital', row['marital'], mapping) if row['marital'] else 'unknown'
    educ = map_val2lab('educ', row['educ'], mapping) if row['educ'] else 'unknown'
    income = map_val2lab('income', row['income'], mapping) if row['income'] else 'unknown'
    wrkstat = map_val2lab('wrkstat', row['wrkstat'], mapping) if row['wrkstat'] else 'unknown'
    occ = map_val2lab('occ10', row['occ10'], mapping) if row['occ10'] else 'unknown'
    hhtype = map_val2lab('hhtype1',row['hhtype1'], mapping) if row['hhtype1'] else 'unknown'

    hh_description = "This person is " + marital +  ", has an education level of " + educ + ", "\
                 "earns an income of " + income + ", is currently " + wrkstat + ", " \
                 "works as a " + occ + ", and belongs to a household type of " + hhtype + "."
    # 
    #prediction
    polviews = row['nateduc']
    pred = "Government spending on education"

    database_background = "The General Social Survey (GSS), conducted since 1972 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.\n"
    role_play_info = f'''
    You are a statistician and a social survey expert. I will give you some information about this person's household information in GSS2022 dataset, 
    which requires you to accurately analyze this person's political views towards {pred}.
      It's a sevenseven-point scale arranging from extremely liberal--point 1--to extremely conservative--point 7.\n "
    '''
    role_play_info = role_play_info + "Your response should consist of just one number (from 1 to 7) to reflect the person's attitude, without any additional text, explanation or even a space letter or a dot. For example, if you think this person has an extremely liberal view, you should response JUST 1 \n"

    return database_background, role_play_info, hh_description, polviews
def extract_float_string(conversation):
    # 匹配包含可选千位分隔符和小数点的数字
    pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?'
    match = re.search(pattern, conversation)
    
    if match:
        # 提取匹配的字符串
        number_str = match.group(0)
        # 去掉逗号以便转换为 float
        number_str = number_str.replace(',', '')
        return float(number_str)
    return None

def main():
    folder_path = './Data/GSS/'
    input_path = '/home/cyyuan/ACL2025/Data/GSS/GSS2022.csv'
    output_path = folder_path + 'GSS_descriptions.txt'
    df = pd.read_csv(input_path) 
    responses1 = []
    responses2 = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for row in reader:

            #hh_description,gts = hh_info(row)
            database_background, role_play_info, hh_description, gt = hh_info(row)
            #无预测值则继续
            if gt == "":
                continue
            cnt = cnt+1
            prompt = database_background+role_play_info+hh_description

            # print(hh_description)
            # print(gt)
            conversation_history = [{"role": "user", "content": prompt}]
            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue       
            print(cnt)

            gt = float(gt)
            gts.append(gt)
            responses1.append(response1)
            responses2.append(response2)
            #print("cnt: ", cnt, " response: ",response1," gts: ",gt)
            
            if cnt==1000:
                break

    responses1 = np.array(responses1)  
    responses2 = np.array(responses2)
    gts = np.array(gts) 
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0] != gts.shape[0]:  
        print(responses1.shape[0])
        print(responses2.shape[0])
        print(gts.shape[0])
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    
    data = np.column_stack((responses1, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/GSS/duolungpt/educ_round3_zero3gpt3.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into educ_round2_zero_gpt3.csv")

    data = np.column_stack((responses2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/GSS/duolungpt/educ_round3_zerogpt4.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into educ_round2_zero_gpt4.csv")



    responses1 = []
    responses2 = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for row in reader:

            #hh_description,gts = hh_info(row)
            database_background, role_play_info, hh_description, gt = hh_info(row)
            #无预测值则继续
            if gt == "":
                continue
            cnt = cnt+1
            prompt = database_background+role_play_info+hh_description+few_examples

            conversation_history = [{"role": "user", "content": prompt}]
            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue       
            print(cnt)

            gt = float(gt)
            gts.append(gt)
            responses1.append(response1)
            responses2.append(response2)
            #print("cnt: ", cnt, " response: ",response1," gts: ",gt)
            
            if cnt==1000:
                break

    responses1 = np.array(responses1)  
    responses2 = np.array(responses2)
    gts = np.array(gts) 
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    
    data = np.column_stack((responses1, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/GSS/duolungpt/educ_round2_few_gpt3.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into educ_round2_few30.csv")

    data = np.column_stack((responses2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/GSS/duolungpt/educ_round2_few_gpt4.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into educ_round2_few31.csv")

if __name__ == "__main__":
    main()

