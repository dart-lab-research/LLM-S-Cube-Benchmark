import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys

from openai import OpenAI
apik = "YOUR_API_KEY"  
Mapping = "D://School//UM//LLM//RECS//mappings//mapping.json"
# 调用 GPT-3.5 模型  
client = OpenAI(api_key=apik)
def ask_gpt(client, messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content

def ask_gpt4(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content


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
        model='llama3.1',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def generate_background():

    database_background = '''
    The Residential Energy Consumption Survey (RECS), conducted by the U.S. Energy Information Administration (EIA), is a nationally representative study designed to collect detailed information on household energy usage, expenditures, and related demographics. 
    Since its inception in 1978, RECS has played a vital role in providing insights into energy consumption patterns across various housing units in the United States. 
    The 2020 RECS dataset, for instance, surveyed over 5,600 households, representing approximately 118.2 million primary residences. 
    The survey encompasses data on energy sources such as electricity, natural gas, propane, and fuel oil, along with expenditure estimates for heating, cooling, and other end uses. 
    This dataset has been instrumental in energy policy analysis, efficiency improvements, and forecasting future consumption trends, making it a cornerstone of energy research in the residential sector.
    '''

#         The distribution ratios of the options selected by respondents on some other questions are as follows:
# The resources the country spends on the military, armaments and defense:
# [1: 0.285, 2: 0.389, 3: 0.325]
# The resources the country spends on halting the rising crime rate:
# [1: 0.726, 2: 0.196, 3: 0.078]
# The resources the country spends on improving the nation's education system:
# [1: 0.75, 2: 0.186, 3: 0.064]
    role_play_info = '''
   You are a data scientist and socioeconomic analyst. Your task is to generate a synthetic dataset simulating residential energy consumption and household characteristics based on general knowledge and logical assumptions. 
    No prior distributional information or feature statistics will be provided. 
    Use your expertise to model realistic energy usage patterns, household demographics, and expenditure data for a nationally representative set of housing units.: '''
    task = f'''
Based on your understanding, prior knowledge, and general reasoning, predict the following for a household:
    1. What is the total electricity use (KWH) of this household in 2020?
    (KWH: Total electricity use, in kilowatt-hours, including self-generation of solar power.)
    2. What is the total electricity cost (DOLLAREL) of this household in 2020?
    (DOLLAREL: Total electricity cost, in dollars.)
    3. What is the total energy cost (TOTALDOL) of this household in 2020?
    (TOTALDOL: Total cost including electricity, natural gas, propane, and fuel oil, in dollars.)
    Generate 50 rows of data in each batch, and repeat this process for 100 batches to simulate a comprehensive dataset.
    For each batch, provide the predictions directly in the following format:

    ['KWH', 'DOLLAREL', 'TOTALODL']
    Here are some examples:
        ['12521', '1955.06', '2656.89'],  
        ['5243', '713.27', '975'],  
        ['2387', '334.51', '522.65'],  
        ['9275', '1424.86', '2061.77'],  
        ['5869', '1087', '1463.04'],  
    After completing a batch, explicitly indicate the batch number before moving to the next batch (e.g., "Batch 1 complete"). Continue this process until all 100 batches are generated.
    The format must be a JSON string representing a three-dimensional array. Also, make sure that it is an array of arrays with no objects, like in a spreadsheet.
    After generating, only show the data you generated without additional words. Remember, the records should closely reflect the RECS dataset.
    '''
    return role_play_info+database_background+task
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/GSS/'
    input_path = folder_path + 'selected_GSS.csv'
    output_path = r"/home/cyyuan/RECS/ol3zero.txt"
    responses = []
    gts = []
    with open(output_path, mode='w', newline='') as outfile:
        
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        prompt = generate_background()
        print(prompt)
        messages = [{"role": "user", "content": prompt}]
        for i in range(20):
           
            print(f"No:{i} generating...")
            aa = ol3( messages)
            messages.append({"role": "assistant", "content": aa})
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 50 times 20
            
        print("done!")
        


    output_path = r"/home/cyyuan/RECS/ol30zero.txt"
    sys.exit(0)
    responses = []
    gts = []
    with open(output_path, mode='w', newline='') as outfile:
        
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        prompt = generate_background()
        print(prompt)
        for i in range(20):
    
            # outfile.write(prompt+'\n')
            messages = [{"role": "user", "content": prompt}]
            
            aa = ol3(messages)
            print(f"No:{i} generating...")
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 20 times 20
            
        print("done!")
if __name__ == "__main__":
    main()