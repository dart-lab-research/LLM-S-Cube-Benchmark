import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from prompt_a20 import pp, load_mapping, base_info, assess

def ol(messages):
    stream = ollama.chat(
        model='llama3.1:latest',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def map_score2option(a): #turn to options
    a = float(a)
    if 0 <= a < 20:
        return 'A'
    elif 20 <= a < 40:
        return 'B'
    elif 40 <= a < 60:
        return 'C'
    elif 60 <= a < 80:
        return 'D'
    elif 80 <= a <= 100:
        return 'E'
    else:
        return 'F'
    
#     Here are some examples and answers:
# This person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Mike Pence.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Andrew Yang. , while at the same time has a neutral feeling (score 60.0, in range of [40,60]) for rating to Clarence Thomas. , has a neutral feeling (score 60.0, in range of [40,60]) for rating to Nancy Pelosi.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Dr. Anthony Fauci. .
# Besides, this person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Donald Trump.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Barack Obama. 
# The answer is D

# This person has a bit warm or favorable feeling (score 65.0, in range of [60,80]) for rating to Mike Pence.  and has a bit warm or favorable feeling (score 70.0, in range of [60,80]) for rating to Andrew Yang. , while at the same time has a bit cold or unfavorable feeling (score 40.0, in range of [20,40]) for rating to Clarence Thomas. , has a neutral feeling (score 50.0, in range of [40,60]) for rating to Nancy Pelosi.  and has a bit warm or favorable feeling (score 80.0, in range of [60,80]) for rating to Dr. Anthony Fauci. .
# Besides, this person has a cold or unfavorable feeling (score 0.0, in range of [0,20]) for rating to Donald Trump.  and has a bit cold or unfavorable feeling (score 40.0, in range of [20,40]) for rating to Barack Obama. 
# The answer is C
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

#####修改这里预测数据
    pred = "Joe Biden"
    pred_value = map_score2option(row['ftbiden1'])

    assess_description = f'''This person {pence} and {yang}, while at the same time {thomas}, {pelosi} and {fauci}.
Besides, this person {trump} and {obama}
'''
    
    roleplay_info = '''The data comes from American National Election Studies 2020 (ANES2020). 
As a sociologist and political scientist, you need to analyze the public opinion of citizens in this dataset and predict their possible election attitudes.

'''

    question = f'''According to the information, can you help me decide what's this person's feeling for rating to {pred}?  
        A. A cold or unfavorable feeling.
        B. A bit cold or unfavorable feeling. 
        C. A a neutral feeling.
        D. A bit warm or favorable feeling
        E. A quite warm or favorable feeling

Your response should consist of just the option A,B,C,D or E to reflect the person's attitude, without any additional text, explanation or even a space letter.
Here is an example of a required response that you should follow: 
    if you think this person has a bit cold or unfavorable feeling, you should response just B (Not B.)
    if you think this person has a quite warm or favorable feeling, you should response just E (Not E.)
'''
    return assess_description, roleplay_info, question,pred_value

def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Anes2020/'
    input_path = folder_path + 'selected_anes2020.csv'
    output_path = folder_path + 'A20chosen_descriptions.txt'
    df = pd.read_csv(input_path) 


    responses = []
    gts = []


    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            assess_description, roleplay_info, question, gt = assessment(row)

            prompt = base_description + assess_description + roleplay_info + question
            conversation_history = [{"role": "user", "content": prompt}]
            response = ol(conversation_history)[0]#他有时候回复A.
            

            responses.append(response)
            gts.append(gt)
            
            if response == gt:
                correct = correct+1

            if i<=10:
                print(f"No:{i}")
                print("response: ",response)
                print("gt: ",gt)
                outfile.write(prompt)
                outfile.write(f"response: {response}\n")
                outfile.write(f"{gt}")
            else:
                print(f"No:{i}")

    acc = correct/i
    print("total num: ", i, "\nacc: ", acc)

    responses = np.array(responses)  
    gts = np.array(gts)  

    #确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + 'responses/chosen_biden3.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into bd.csv")

if __name__ == "__main__":
    main()