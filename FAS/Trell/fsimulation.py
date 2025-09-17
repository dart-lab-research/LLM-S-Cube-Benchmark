import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys

from openai import OpenAI
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


def ol(messages):

    stream = ollama.chat(
        model='llama3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

# The distribution ratios of the options selected by respondents on some other questions are as follows:
# The resources the country spends on the military, armaments and defense:
# [1: 0.285, 2: 0.389, 3: 0.325]
# The resources the country spends on halting the rising crime rate:
# [1: 0.726, 2: 0.196, 3: 0.078]
# The resources the country spends on improving the nation's education system:
# [1: 0.75, 2: 0.186, 3: 0.064]
    
def generate_background():

    database_background = '''
    The General Social Survey 2020 (GSS2020), conducted in 2020 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. 
    Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.
    '''


    role_play_info = '''
   You are a statistician and a social survey expert. Your task is to generate a survey dataset: '''
    task = f'''
The sample batch size is 20. The dataset should include 3 demographic variables for a respondent, whose attitude towards the following questions: 

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
    output_path = r"D:\School\UM\LLM\Media\35zero.txt"
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
            print(f"No:{i} generating...")
            aa = ask_gpt(client, messages)
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 20 times 20
            if i >= 20:
                print("Done! ")
                break
        
if __name__ == "__main__":
    main()