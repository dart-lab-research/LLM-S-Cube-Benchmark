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
   You are a data scientist and socioeconomic analyst. I will provide you with the demographic distribution of a population, including the proportions of gender (HHSEX), age groups (HHAGE), employment status (EMPLOYHH), and state location (state_postal). 
Based on these distributions, your task is to generate a synthetic dataset that reflects the specified demographic structure. 
The generated dataset should include plausible and consistent combinations of features that align with the given proportions, simulating a realistic social survey dataset. Ensure that the dataset captures meaningful patterns and relationships between features commonly observed in real-world surveys.
'''
    task = f'''
Based on your understanding, prior knowledge, and general reasoning, predict the following for a household:
    1. What is the total electricity use (KWH) of this household in 2020?
    (KWH: Total electricity use, in kilowatt-hours, including self-generation of solar power.)
    2. What is the total electricity cost (DOLLAREL) of this household in 2020?
    (DOLLAREL: Total electricity cost, in dollars.)
    3. What is the total energy cost (TOTALDOL) of this household in 2020?
    (TOTALDOL: Total cost including electricity, natural gas, propane, and fuel oil, in dollars.)
    Generate 50 rows of data in each batch, and repeat this process for 100 batches to simulate a comprehensive dataset.

    I will inform you in advance about the distribution of all respondents by gender, age group, employment status and state postal. As follows:
        HHSEX: Respondent sex 
        [Female: 0.5408, Male: 0.45.92]

        HHAGE: Respondent age group 
        [18-30: 0.0868, 31-40: 0.1503, 41-50: 0.1496, 51-60: 0.1848, 61-70: 0.2087, 71-80: 0.1574, 81-90: 0.0623]

        EMPLOYHH: Respondent employment status 
        [Employed full-time: 0.4706, Employed part-time: 0.0868, Not employed: 0.1166, Retired:0.3261]

        state_postal:
            Alabama: 0.0131,  
            Alaska: 0.0168, 
            Arizona: 0.0268,  
            Arkansas: 0.0145,  
            California: 0.0623,  
            Colorado: 0.0195,  
            Connecticut: 0.0159,  
            Delaware: 0.0077,  
            District of Columbia: 0.0119,  
            Florida: 0.0354,  
            Georgia: 0.0225,  
            Hawaii: 0.0152,  
            Idaho: 0.0146,  
            Illinois: 0.0287,  
            Indiana: 0.0216,  
            Iowa: 0.0155,  
            Kansas: 0.0112,  
            Kentucky: 0.0231,  
            Louisiana: 0.0168,  
            Maine: 0.0121,  
            Maryland: 0.0194,  
            Massachusetts: 0.0298,  
            Michigan: 0.0210,  
            Minnesota: 0.0176,  
            Mississippi: 0.0091,  
            Missouri: 0.0160, 
            Montana: 0.0093,  
            Nebraska: 0.0102,  
            Nevada: 0.0125,  
            New Hampshire: 0.0095,  
            New Jersey: 0.0247,  
            New Mexico: 0.0096,  
            New York: 0.0489,  
            North Carolina: 0.0259,  
            North Dakota: 0.0179,  
            Ohio: 0.0183,  
            Oklahoma: 0.0125,  
            Oregon: 0.0169,  
            Pennsylvania: 0.0334,  
            Rhode Island: 0.0103,  
            South Carolina: 0.0181,  
            South Dakota: 0.0099,  
            Tennessee: 0.0273,  
            Texas: 0.0549,  
            Utah: 0.0102,  
            Vermont: 0.0132,  
            Virginia: 0.0244,  
            Washington: 0.0237,  
            West Virginia: 0.0107,  
            Wisconsin: 0.0193,  
            Wyoming: 0.0103

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
    output_path = r"/home/cyyuan/RECS/gpt35few.txt"
    responses = []
    gts = []
    with open(output_path, mode='w', newline='') as outfile:
        
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        prompt = generate_background()
        messages = [{"role": "user", "content": prompt}]
        print(prompt)
        for i in range(20):
    
            # outfile.write(prompt+'\n')
            
            print(f"No:{i} generating...")
            #aa = ol31( messages)
            aa = ask_gpt(client, messages)
            messages.append({"role": "assistant", "content": aa})
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 50 times 20
            
        print("done!")
        


    output_path = r"/home/cyyuan/RECS/ol3few.txt"
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
            
            aa = ask_gpt(client, messages)
            print(f"No:{i} generating...")
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 20 times 20
            
        print("done!")
if __name__ == "__main__":
    main()