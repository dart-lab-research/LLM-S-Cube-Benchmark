import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from transformers import pipeline
import torch

#支持情况
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
#####修改这里预测数据
    pred = "Donald Trump"
    pred_value = row['fttrump1']
    assess_description = f'''This person {pence} and {yang}, while at the same time {thomas} and {fauci}.
Besides, this person {biden} and {trump}
'''
    
    roleplay_info = '''The data comes from American National Election Studies 2020 (ANES2020). 
As a sociologist and political scientist, you need to analyze the public opinion of citizens in this dataset and predict their possible election attitudes.

'''

    question = f'''According to the information, can you help me decide how would this person rate {obama}?  
Your response should consist of just one number between [0, 100] to reflect the person's attitude, without any additional text, explanation or even a space letter.
Here is an example of a required response that you should follow: 
    if you think this person has a fairly cold or unfavorable feeling, you should response JUST a integer number like 25. (or more/less favorable you think, the number may be bigger/smaller)
So your response should be like 25
'''
    

    return assess_description, roleplay_info, question, pred_value
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Anes2020/'
    input_path = '/home/cyyuan/ACL2025/Data/Anes2020/selected_anes2020.csv'
    output_path = folder_path + 'a20_descriptions.txt'
    df = pd.read_csv(input_path) 


    responses = []
    gts = []


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            assess_description, roleplayinfo, question ,gt = assessment(row)

            prompt = base_description + assess_description +roleplayinfo + question
            conversation_history = [{"role": "user", "content": prompt}]

            response = ol(conversation_history)
            responses.append(response)
            gts.append(float(gt))
            
            print(f"No:{i}")

            
            # print(assess_description, roleplayinfo, question)
            # print(predvalue)
            #  
            
            # conversation_history = [{"role": "user", "content": prompt}]
            # response = float(ol(conversation_history))

             
    responses = np.array(responses)  
    gts = np.array(gts)  

    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/Anes2020/duolun/obama3zero_round2.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into obm30.csv")
            
    
     

    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            assess_description, roleplayinfo, question ,gt = assessment(row)

            prompt = base_description + assess_description +roleplayinfo + question
            conversation_history = [{"role": "user", "content": prompt}]
            response = float(ol31(conversation_history))
            responses.append(response)
            gts.append(float(gt))
            
            print(f"No:{i}")

            
            # print(assess_description, roleplayinfo, question)
            # print(predvalue)
            #  
            
            # conversation_history = [{"role": "user", "content": prompt}]
            # response = float(ol(conversation_history))
    responses = np.array(responses)  
    gts = np.array(gts)  

    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/Anes2020/duolun/obama31zero_round2.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into obm31.csv")
if __name__ == "__main__":
    main()