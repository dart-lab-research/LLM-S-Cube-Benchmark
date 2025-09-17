
import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from transformers import pipeline
import torch
import re
from openai import OpenAI
from utils import apik

client = OpenAI(api_key=apik)
def ask_gpt3(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
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



#支持情况
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

def ol(messages):
    stream = ollama.chat(
        model='llama3:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def dpsk(messages):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",device= device)
    response = pipe(messages)

    assistant_response = response[0]['generated_text'][-1]['content']
    return assistant_response

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
def load_mapping(file_path):  
    with open(file_path, 'r') as file:  
        return json.load(file)  

def pp(value):
    if value == '0':
        return 'This person has not '
    else: 
        return 'This person has '
def assess(value):
    value = float(value)
    if value <= 20:
        return f'has a cold or unfavorable feeling (score {value}, in range of [0,20]) for rating to '
    if value>20 and value<=40:
        return f'has a bit cold or unfavorable feeling (score {value}, in range of [20,40]) for rating to '
    if value>40 and value <=60:
        return f'has a neutral feeling (score {value}, in range of [40,60]) for rating to '
    if value>60 and value<=80:
        return f'has a bit warm or favorable feeling (score {value}, in range of [60,80]) for rating to '
    if value > 80 and value != 999:
        return f'has a quite warm or favorable feeling for (score {value}, in range of [80,100]) rating to '
    else:
        return f'has an unknown feeling for rating (score {value}) to '
    
def base_info(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/Anes2020/mappings/mapping.json')  

    age = row['age']
    hh_ownership =  mapping['home_ownership'].get(str(int(row['home_ownership'])))  
    income = mapping['income'].get(str(int(row['income'])))
    vote = mapping['vote20turnoutjb'].get(str(int(row['vote20turnoutjb'])))  
    score = row['particip_count']

    base_description = f'''
This person, who ages {age}, {hh_ownership} and has an income between {income}. Besides, this person has a voting tendency: {vote}. 
This person has a summarized participation score {score}.

'''
    return base_description



# Here are some examples and answers:
# 1. This person, who ages 69, homeowner and has an income between $10,000 - $14,999. Besides, this person has a voting tendency: Probably would not vote. 
# This person has a summarized participation score 0.
# This person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Mike Pence.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Andrew Yang. , while at the same time has a neutral feeling (score 60.0, in range of [40,60]) for rating to Clarence Thomas.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Dr. Anthony Fauci. .
# Besides, this person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Joe Biden.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Donald Trump. 
# The answer is 60

# 2. This person, who ages 22, homeowner and has an income between $75,000 - $79,999. Besides, this person has a voting tendency: Select someone to vote for. 
# This person has a summarized participation score 5.
# This person has a bit warm or favorable feeling (score 65.0, in range of [60,80]) for rating to Mike Pence.  and has a bit warm or favorable feeling (score 70.0, in range of [60,80]) for rating to Andrew Yang. , while at the same time has a bit cold or unfavorable feeling (score 40.0, in range of [20,40]) for rating to Clarence Thomas.  and has a bit warm or favorable feeling (score 80.0, in range of [60,80]) for rating to Dr. Anthony Fauci. .
# Besides, this person has a neutral feeling (score 50.0, in range of [40,60]) for rating to Joe Biden.  and has a cold or unfavorable feeling (score 0.0, in range of [0,20]) for rating to Donald Trump. 
# The answer is 40
def assessment(row):
    #politician_features = ['ftpence1', 'ftyang1', 'ftpelosi1', 'ftrubio1','ftocasioc1', 'fthaley1','ftthomas1', 'ftfauci1']
    pence = assess(row['ftpence1']) + "Mike Pence. "
    yang = assess(row['ftyang1']) + "Andrew Yang. "
    pelosi = assess(row['ftpelosi1']) + "Nancy Pelosi. "
    rubio = assess(row['ftrubio1']) + "Marco Rubio. "
    ocasioc = assess(row['ftocasioc1']) + "Alexandria Ocasio-Cortez. "
    haley = assess(row['fthaley1'])+ "Nikki Haley. "
    thomas = assess(row['ftthomas1']) + "Clarence Thomas. "
    fauci = assess(row['ftfauci1']) + "Dr. Anthony Fauci. "


    #president
    trump = assess(row['fttrump1']) + "Donald Trump. "
    obama = assess(row['ftobama1'])+ "Barack Obama. "
    biden = assess(row['ftbiden1'])+ "Joe Biden. "

    pred = "Donald Trump"
    pred_value = row['fttrump1']
    assess_description = f'''This person {pence} and {yang}, while at the same time {thomas} and {fauci}.
Besides, this person {biden} and {obama}
'''
    
    roleplay_info = '''The data comes from American National Election Studies 2020 (ANES2020). 
As a sociologist and political scientist, you need to analyze the public opinion of citizens in this dataset and predict their possible election attitudes.

'''

    question = f'''According to the information, can you help me decide how would this person rate {pred}?  
Your response should consist of just one number between [0, 100] to reflect the person's attitude, without any additional text, explanation or even a space letter.
Here is an example of a required response that you should follow: 
    if you think this person has a fairly cold or unfavorable feeling, you should response JUST a integer number like 25. (or more/less favorable you think, the number may be bigger/smaller)
So your response should be like 25
'''
    

    return assess_description, roleplay_info, question, pred_value

few_examples = '''
Here are some examples and answers:
1. This person, who ages 69, homeowner and has an income between $10,000 - $14,999. Besides, this person has a voting tendency: Probably would not vote. 
This person has a summarized participation score 0.
This person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Mike Pence.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Andrew Yang. , while at the same time has a neutral feeling (score 60.0, in range of [40,60]) for rating to Clarence Thomas.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Dr. Anthony Fauci. .
The answer is 60

2. This person, who ages 22, homeowner and has an income between $75,000 - $79,999. Besides, this person has a voting tendency: Select someone to vote for. 
This person has a summarized participation score 5.
This person has a bit warm or favorable feeling (score 65.0, in range of [60,80]) for rating to Mike Pence.  and has a bit warm or favorable feeling (score 70.0, in range of [60,80]) for rating to Andrew Yang. , while at the same time has a bit cold or unfavorable feeling (score 40.0, in range of [20,40]) for rating to Clarence Thomas.  and has a bit warm or favorable feeling (score 80.0, in range of [60,80]) for rating to Dr. Anthony Fauci. .
The answer is 40
'''
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Anes2020/'
    input_path = '/home/cyyuan/ACL2025/Data/Anes2020/selected_anes2020.csv'
    output_path = folder_path + 'a20_descriptions.txt'
    log_path = '/home/cyyuan/ACL2025/Anes2020/gpt/log.txt'
    df = pd.read_csv(input_path) 


    responses = []
    gts = []


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        responses1 = []
        responses2 = []
        responsesfew1 = []
        responsesfew2 = []
        with open('/home/cyyuan/ACL2025/Data/Anes2020/duolungpt/trump_round3_inrow.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入标题行
                writer.writerow(['gpt3.5_zero','gpt4_zero','gpt3.5few','gpt4few','gts'])
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            assess_description, roleplayinfo, question ,gt = assessment(row)

            prompt = base_description + assess_description +roleplayinfo + question
            conversation_history = [{"role": "user", "content": prompt}]
            conversation_few = [{"role":"user","content":prompt+few_examples}]
            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
                responsefew1 = extract_float_string(ask_gpt3(client,conversation_few))
                responsefew2 = extract_float_string(ask_gpt4(client,conversation_few))
                # print(response1)
                # print(response2)
            except (ValueError, TypeError) as e:
            # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}/{conversation_few}, error: {e}")
                with open(log_path,"a") as file:
                    file.write(f"id:{id}:errortype:{e}\n")
                continue
            
            with open('/home/cyyuan/ACL2025/Data/Anes2020/duolungpt/trump_round3_inrow.csv', mode='a', newline='') as file:
                # 写入数据行
                writer = csv.writer(file)
                writer.writerow([response1, response2,responsefew1,responsefew2,gt])
                print(f"write already, No:{i}")
            
            responses1.append(response1)
            responses2.append(response2)
            responsesfew2.append(responsefew2)
            responsesfew1.append(responsefew1)
            gts.append(float(gt))
            print(f"No:{i}")
            #print(prompt)
            if i==3000:
                break       

    responses1 = np.array(responses1)  
    responses2 = np.array(responses2)
    responsesfew1 = np.array(responsesfew1)
    responsesfew2 = np.array(responsesfew2)
    gts = np.array(gts)  

    # 确保两个数组的长度相同  
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0] != gts.shape[0] or responsesfew1.shape[0] != gts.shape[0] or responsesfew2.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses1,responses2,responsesfew1,responsesfew2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/Anes2020/duolungpt/trump_round3.csv', data, delimiter=',', header='gpt3.5_zero,gpt4_zero,gpt3.5few,gpt4few,gts', comments='', fmt='%s')  
    print("complete writing data into trump35.csv")
            
if __name__ == "__main__":
    main()
