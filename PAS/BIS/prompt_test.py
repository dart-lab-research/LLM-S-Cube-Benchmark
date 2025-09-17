import pandas as pd
import csv
import json
import ollama
import numpy as np
import re
from openai import OpenAI
from utils import apik

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

def load_mapping(file_path):  
    with open(file_path, 'r') as file:  
        return json.load(file)  
    
def ol3(messages):
    stream = ollama.chat(
        model='llama3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans
def ol31(messages):
    stream = ollama.chat(
        model='llama3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans
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
def generate_baseinfo(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/Income/mappings/mapping.json')  
    uuid = row['uuid']
    age = row['age']
    gender = mapping["gender"].get(row["gender"])  
    rural = mapping["rural"].get(row["rural"])  
    dem_education_level = mapping['dem_education_level'].get(row['dem_education_level'])
    dem_full_time_job = mapping['dem_full_time_job'].get(row['dem_full_time_job'])
    dem_has_children = mapping['dem_has_children'].get(row['dem_has_children'])

    base_info = f'''
This person, identified by {uuid}, is a {age}-year-old {gender}, {rural} {dem_education_level}, {dem_full_time_job}, and {dem_has_children}. 
'''
    return base_info

def generate_description(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/Income/mappings/mapping.json')  

    question_bbi_2016wave4_basicincome_awareness = mapping['question_bbi_2016wave4_basicincome_awareness'].get(row['question_bbi_2016wave4_basicincome_awareness'])
    question_bbi_2016wave4_basicincome_vote = mapping['question_bbi_2016wave4_basicincome_vote'].get(row['question_bbi_2016wave4_basicincome_vote'])
    question_bbi_2016wave4_basicincome_effect = mapping['question_bbi_2016wave4_basicincome_effect'].get(row['question_bbi_2016wave4_basicincome_effect'])
    question_bbi_2016wave4_basicincome_argumentsfor = mapping['question_bbi_2016wave4_basicincome_argumentsfor'].get(row['question_bbi_2016wave4_basicincome_argumentsfor'])
    question_bbi_2016wave4_basicincome_argumentsagainst = mapping['question_bbi_2016wave4_basicincome_argumentsagainst'].get(row['question_bbi_2016wave4_basicincome_argumentsagainst'])

    # Question 3: What could be the most likely effect of basic income on your work choices? I would...
    # The individual's answer is {question_bbi_2016wave4_basicincome_effect}
    # Question 2: Which of the following arguments FOR the basic income do you find convincing?
    # The individual's answer is {question_bbi_2016wave4_basicincome_argumentsfor}
    que = f'''how would this person vote if there would be a referendum on introducing basic income today
'''
    part_responses = f'''
Their responses to the survey questions are as follows:
Question 1: How familiar are you with the concept known as "basic income"
The individual's answer is {question_bbi_2016wave4_basicincome_awareness}
additional information about basic income: "A basic income is an income unconditionally paid by the government to every individual regardless of whether they work and irrespective of any other sources of income. It replaces other social security payments and is high enough to cover all basic needs (food, housing etc.)."

Question 2: Which of the following arguments FOR the basic income do you find convincing?
The individual's answer is {question_bbi_2016wave4_basicincome_argumentsfor}

Question 3: What could be the most likely effect of basic income on your work choices?
The individual's answer is {question_bbi_2016wave4_basicincome_effect}
'''

    roleplay_setting = f'''
You are a statistician and a social survey expert. I will give you some information about this person's demographic background and their responses to several questions regarding basic income in the basic_income dataset, 
which requires you to accurately analyze this person's perspective and predict {que}.
'''
    background = '''
This dataset is based on a European public opinion study on basic income conducted by Dalia Research in April 2016 across 28 EU Member States. 
The survey included 9,649 participants aged 14-65, with the sample designed to reflect population distributions for age, gender, region/country, education level (ISCED 2011), and urbanization (rural vs. urban).

'''

    return background, part_responses, roleplay_setting

def generate_questions(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/Income/mappings/mapping.json')  
    question_bbi_2016wave4_basicincome_argumentsfor = mapping['question_bbi_2016wave4_basicincome_argumentsfor'].get(row['question_bbi_2016wave4_basicincome_argumentsfor'])
    
    
    #更改pred处
    #pred = mapping['answer_bbi_2016wave4_basicincome_argumentsagainst'].get(row['question_bbi_2016wave4_basicincome_argumentsagainst'])
    pred = mapping['answer_bbi_2016wave4_basicincome_vote'].get(row['question_bbi_2016wave4_basicincome_vote'])
    #pred = mapping['answer_bbi_2016wave4_basicincome_effect'].get(row['question_bbi_2016wave4_basicincome_effect'])
    #pred = mapping['answer_bbi_2016wave4_basicincome_argumentsfor'].get(row['question_bbi_2016wave4_basicincome_argumentsfor'])
    que = f'''how would this person vote if there would be a referendum on introducing basic income today
'''
    prediction_task = f'''
    Predict {que}, using the predefined answer options below.
    
    Answer options:
    1. I would vote for it
    2. I would probably vote for it
    3. I would probably vote against it
    4. I would vote against it
    5. I would not vote

    Your response should consist of just the option (1/2/3/4/5) to reflect the person's opinion, without any additional text, explanation or even a space letter. You are not allowed to explain anything of your response. Your entire output should not exceed 1 character.
    If your answer is 1, your response should just be 1, without any additional text or explanation.
    If your answer is 3, your response should just be 3, without any additional text or explanation.    

    '''
    return prediction_task, pred,row['uuid']

few_examples = '''
Here are some examples and answers:

1. This person, identified by f6e7ee00-deac-0133-4de8-0a81e8b09a82, is a 61-year-old male., This person lives in a rural area. This person doesn't have a formal education., This person does not have a full-time job., and There are no children in the individual's current household.. 
The answer is 5

2. This person, identified by 54f0f1c0-dda1-0133-a559-0a81e8b09a82, is a 57-year-old male., This person lives in an urban area. This person has a high level of formal education., This person has a full-time job., and There are children in the individual's current household.. 
The answer is 2
'''
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
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Income/'
    input_path = '/home/cyyuan/ACL2025/Data/Income/basic_income_dataset_dalia.csv'
    output_path = folder_path + 'income_descriptions_vote.txt'
    log_path = '/home/cyyuan/ACL2025/Data/Income/duolun/log.txt'
    df = pd.read_csv(input_path) 

    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        for i,row in enumerate(reader):
            background, part_response, roleplay_setting = generate_description(row)
            baseinfo = generate_baseinfo(row)
            pred_task,gt,id = generate_questions(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
            
            print("ground truth: ", gt)
            prompt = background + baseinfo + part_response + roleplay_setting+ pred_task
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                with open(log_path,"a") as file:
                    file.write(f"No:{id},error type:{e}\n")
                continue 
            
            if response1 == gt:
                correct1 = correct1+1
            if response2 == gt:
                correct2 = correct2+1
            print(f"No:{i}")
            
            if i == 1000:
                break
    cnt = i - cnt + 1
    acc1 = correct1/cnt
    acc2 = correct2/cnt
    acc1 = format(acc1*100,".2f")
    acc2 = format(acc2*100,".2f")

    with open('/home/cyyuan/ACL2025/Data/Income/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"VOTE_round3_zero_gpt3:  {acc1}%\n")
        file.write(f"VOTE_round3_zero_gpt4:  {acc2}%\n")

    print("write successfully")


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        for i,row in enumerate(reader):
            background, part_response, roleplay_setting = generate_description(row)
            baseinfo = generate_baseinfo(row)
            pred_task,gt,id = generate_questions(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
            
            print("ground truth: ", gt)
            prompt = background + baseinfo + part_response + roleplay_setting+ pred_task + few_examples
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            
            if response1 == gt:
                correct1 = correct1+1
            if response2 == gt:
                correct2 = correct2+1
            print(f"No:{i}")

            if i == 1000:
                break

    cnt = i - cnt + 1
    acc1 = correct1/cnt
    acc2 = correct2/cnt
    acc1 = format(acc1*100,".2f")
    acc2 = format(acc2*100,".2f")

    with open('/home/cyyuan/ACL2025/Data/Income/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"VOTE_round3_few_gpt3:  {acc1}%\n")
        file.write(f"VOTE_round3_few_gpt4:  {acc2}%\n")

    print("write successfully")
if __name__ == "__main__":
    main()