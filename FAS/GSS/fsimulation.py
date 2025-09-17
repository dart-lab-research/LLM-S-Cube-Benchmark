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
        model="gpt-4-turbo",
        messages=messages,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content


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

def generate_background():

    database_background = '''
    The General Social Survey 2020 (GSS2020), conducted in 2020 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. 
    Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.
    '''

#         The distribution ratios of the options selected by respondents on some other questions are as follows:
# The resources the country spends on the military, armaments and defense:
# [1: 0.285, 2: 0.389, 3: 0.325]
# The resources the country spends on halting the rising crime rate:
# [1: 0.726, 2: 0.196, 3: 0.078]
# The resources the country spends on improving the nation's education system:
# [1: 0.75, 2: 0.186, 3: 0.064]
    role_play_info = '''
   You are a statistician and a social survey expert. Your task is to generate a survey dataset: '''
    task = f'''
The sample batch size is 50. The dataset should include 3 demographic variables for a respondent, whose attitude towards the following questions: 

The resources the country spends on improving and protecting the environment is 
    1.too little. 
    2.the right amount. 
    3.too much. 
The resources the country spends on improving and protecting the nation's health is
    1.too little. 
    2.the right amount. 
    3.too much. 
The resources the country spends on solving the problems of big cities
    1.too little. 
    2.the right amount. 
    3.too much. 
 
    
    Here are some examples of data record list: 
['1', '1', '2']
['1', '2', '1']
['1', '1', '1']
['1', '1', '2']
['1', '1', '1']

    Explanations: ['1', '2', '3'] (which represent the choices of the three questions are "too little","the right amount" and "too much")

    Choose your answers only from the options provided. After generating, only show the data you generated without additional words.
    The format must be a JSON string representing a three-dimensional array. Also, make sure that it is an array of arrays with no objects, like in a spreadsheet.
    Remember, the records should closely reflect the GSS dataset."
'''
    return role_play_info+database_background+task
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/GSS/'
    input_path = folder_path + 'selected_GSS.csv'
    output_path = folder_path + 'testForpic.txt'
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        prompt = generate_background()
        messages = [{"role": "user", "content": prompt}]
        print(prompt)
        for i, row in enumerate(reader):
    
            # outfile.write(prompt+'\n')
            
            print(f"No:{i} generating...")
            aa = ol( messages)
            messages.append({"role": "assistant", "content": aa})
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 20 times 20
            if i >= 20:
                print("Done! ")
                break
        
if __name__ == "__main__":
    main()