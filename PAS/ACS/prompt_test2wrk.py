import pandas as pd
import csv
import json
import ollama
import numpy as np
import os
print("Current working directory:", os.getcwd())
def load_mapping(file_path):  
    with open(file_path, 'r') as file:  
        return json.load(file)  
    
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

def generate_baseinfo(row):
    
    mapping = load_mapping('/home/cyyuan/ACL2025/ACS/mappings/mapping.json')  
    age = row['age']
    race = mapping['Race'].get(row['race'])
    gender = mapping['gender'].get(row['gender'])

    birth_qrtr = mapping['birth_qrtr'].get(row['birth_qrtr'])
    citizen  = mapping['citizen'].get(row['citizen'])
    lang = mapping['lang'].get(row['lang'])
    edu  = mapping['edu'].get(row['edu'])

    married = mapping['married'].get(row['married'])
    disability = mapping['disability'].get(row['disability'])
    hrs_work = row['hrs_work']


    employment = mapping['employment'].get(row['employment'])
    time_to_work = row['time_to_work']
    income = row['income']

    pred = hrs_work
    #annual income of the individual
    task = f'''the average work time per week of the individual'''

    base_description  = f'''
This person is identified by the following demographic characteristics and employment-related information: they are a {age}-year-old {race} {gender}, 
born {birth_qrtr} of the year. {citizen}, who speaks {lang} at home, and have attained an education level of {edu}. 
This individual is {married}, {disability} , and the annual income of the individual is {income}.
The individual's employment status is {employment}, with a commute time of {time_to_work} minutes (if applicable).
    '''
    
    rolepplay = f'''
You are a data scientist and socioeconomic analyst. I will provide you with demographic and employment-related information about an individual from the American Community Survey dataset. 
Based on this information, your task is to predict the {task}, using patterns and relationships derived from the provided features. 
    '''

    question = f'''Based on the respondent's background and responses, predict their answers now.    
    Here are some examples and answers:
    1. This person is identified by the following demographic characteristics and employment-related information: they are a 68-year-old white female,, 
    born between July and September. of the year. The respondent is a U.S. citizen., who speaks English at home, and have attained an education level of college.. 
    This individual is not married, does not have a disability. , and the hours worked per week of the individual is 40.
    The individual's employment status is not in the labor force, with a commute time of NA minutes (if applicable).
    The answer of the average worktime of the individual is 40.

    2. This person is identified by the following demographic characteristics and employment-related information: they are a 27-year-old white male,, 
    born between October and December. of the year. The respondent is a U.S. citizen., who speaks English at home, and have attained an education level of high school or lower.. 
    This individual is married, does not have a disability. , and the 84 hours worked per week.
    The individual's employment status is employed, with a commute time of 40 minutes (if applicable).
    The answer of the average worktime of the individual is 84.

    3. This person is identified by the following demographic characteristics and employment-related information: they are a 27-year-old white male,, 
    born between October and December. of the year. The respondent is a U.S. citizen., who speaks English at home, and have attained an education level of high school or lower.. 
    This individual is married, does not have a disability. , and the hours worked per week of the individual is 84.
    The individual's employment status is employed, with a commute time of 40 minutes (if applicable).
    The answer of the average worktime of the individual is 23.

    4. This person is identified by the following demographic characteristics and employment-related information: they are a 52-year-old white male,, 
    born between April and June. of the year. The respondent is a U.S. citizen., who speaks English at home, and have attained an education level of high school or lower.. 
    This individual is married, does not have a disability. , and the hours worked per week of the individual is 55.
    The individual's employment status is employed, with a commute time of 20 minutes (if applicable).
    The answer of the average worktime of the individual is 40.

    This person is identified by the following demographic characteristics and employment-related information: they are a 67-year-old white female,, 
    born between April and June. of the year. The respondent is a U.S. citizen., who speaks English at home, and have attained an education level of high school or lower.. 
    This individual is married, does not have a disability. , and the hours worked per week of the individual is 8.
    The individual's employment status is employed, with a commute time of 10 minutes (if applicable).
    The answer of the average worktime of the individual is 56.
The predictions should be presented as a concise numerical output, without any additional text or explanations. You don't have to say something like "based on...". Remember, you are not allowed to response more than one numerical number. 
If the predicted {task} is 40h, the output should simply be:
40
'''
    
    ans = f"The answer of {task} is {pred}"
    
    return base_description, rolepplay, question, pred



def generate_description(row):

    background = f'''
The ACS (American Community Survey) dataset provides an in-depth examination of the socioeconomic and demographic landscape in the United States. 
With data encompassing a variety of attributes such as income, employment, education, and demographic factors, the ACS dataset allows for a comprehensive analysis of societal trends and individual characteristics. 
Its versatility makes it suitable for exploring correlations, identifying patterns, and informing policy decisions.
    
    '''
    dataset_bg = f'''
Key features of this dataset include information on U.S. citizenship, language spoken at home, marital status, disabilities, and even birth quarter. 
Together, these variables offer a detailed perspective on the interplay between socioeconomic status and personal demographics, enabling researchers to analyze key questions such as the influence of education on income, the impact of travel time on employment, or the relationship between disability status and hours worked.
'''
    
    return background, dataset_bg

def generate_questions(row):


    return
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/ACS/'
    input_path = folder_path + 'acs.csv'
    input_path = '/home/cyyuan/ACL2025/Data/ACS/acs.csv'
    output_path = folder_path + 'acs_descriptions.txt'
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        
        cnt = 0
        correct = 0
        for i,row in enumerate(reader):
            background, dataset_bg = generate_description(row)
            base_description, rolepplay, question, gt = generate_baseinfo(row)
            
            prompt  = background+dataset_bg+base_description+rolepplay+question
            
            #print(str(gt))
            if str(gt) == 'NA' or str(gt) == '0':
                 
                 cnt = cnt +1
                 continue
            
            conversation_history = [{"role": "user", "content": prompt}]

            try:  
            # 尝试将字符串转换为浮点数  
                response = float(ol3(conversation_history).replace(',', ''))  # 去掉逗号并转换为浮点数  
                print(f"Converted value: {response}")  
            except ValueError:  
            # 如果发生 ValueError，打印错误信息并继续循环  
                print(f"Skipping invalid value: {response}")  

            responses.append(response)
            gts.append(gt)
            
           
            print(f"No:{i}")
            
    print(i-cnt)       
    # cnt = i - cnt
    # acc = correct/cnt
    # print("total num: ", cnt, "\nacc: ", acc)

    responses = np.array(responses)  
    gts = np.array(gts)  
    
    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/ACS/responses/acs_wrkhours30few2.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into acs_income.csv")
            
    
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        
        cnt = 0
        correct = 0
        for i,row in enumerate(reader):
            background, dataset_bg = generate_description(row)
            base_description, rolepplay, question, gt = generate_baseinfo(row)
            
            prompt  = background+dataset_bg+base_description+rolepplay+question
            #print(str(gt))
            if str(gt) == 'NA' or str(gt) == '0':
                 
                 cnt = cnt +1
                 continue
            
            conversation_history = [{"role": "user", "content": prompt}]

            try:  
            # 尝试将字符串转换为浮点数  
                response = float(ol3(conversation_history).replace(',', ''))  # 去掉逗号并转换为浮点数  
                print(f"Converted value: {response}")  
            except ValueError:  
            # 如果发生 ValueError，打印错误信息并继续循环  
                print(f"Skipping invalid value: {response}")  

            responses.append(response)
            gts.append(gt)
            
            print(f"No:{i}")
            

    print(i-cnt)       
    # cnt = i - cnt
    # acc = correct/cnt
    # print("total num: ", cnt, "\nacc: ", acc)

    responses = np.array(responses)  
    gts = np.array(gts)  
    
    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/ACS/responses/acs_wrkhours31few2.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into acs_income.csv")
if __name__ == "__main__":
    main()