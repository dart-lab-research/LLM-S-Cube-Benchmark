import pandas as pd
import csv
import json
import ollama
import numpy as np
import re
from utils import ask_gpt4, ask_gpt3, apik, client
def load_mapping(file_path):  
    with open(file_path, 'r') as file:  
        return json.load(file)  
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
    mapping = load_mapping('/home/cyyuan/ACL2025/RECS/mappings/mapping.json')
    age = row['HHAGE']
    gender = mapping["HHSEX"].get(row["HHSEX"])  
    employment = mapping['EMPLOYHH'].get(row['EMPLOYHH'])
    state = mapping['state_postal'].get(row['state_postal'])
    hhrace = mapping['HOUSEHOLDER_RACE'].get(row['HOUSEHOLDER_RACE'])
    hhmember = row['NHSLDMEM']
    athome = mapping['ATHOME'].get(row['ATHOME'])
    income = mapping['MONEYPY'].get(row['MONEYPY'])

    background = f'''
The Residential Energy Consumption Survey (RECS), conducted by the U.S. Energy Information Administration (EIA), is a nationally representative dataset that provides detailed insights into household energy usage, costs, and characteristics. 
First launched in 1978, the 2020 RECS cycle collected data from nearly 18,500 households, representing 123.5 million primary residence housing units. 
The survey gathers information on energy usage patterns, housing characteristics, and household demographics through web and mail forms, supplemented by data from energy suppliers.
    '''
    base_info = f'''
This person, who is a {age}-year-old {gender}, whose employment status is {employment} and lives in the state {state}.
Besides, this person's race is {hhrace} and the number of household members is {hhmember}, while {athome}.
Moreover, the total household income is {income}.
'''

    return background, base_info

def generate_description(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/RECS/mappings/mapping.json')

    gas = mapping["UGASHERE"].get(row["UGASHERE"])  
    electricity_cooking = mapping["ELFOOD"].get(row["ELFOOD"])  
    propane_cooking = mapping["LPCOOK"].get(row["LPCOOK"])  
    natural_gas_cooking = mapping["UGCOOK"].get(row["UGCOOK"])

    electricity_used = mapping["USEEL"].get(row["USEEL"])  
    natural_gas_used = mapping["USENG"].get(row["USENG"])  
    propane_used = mapping["USELP"].get(row["USELP"])  
    fuel_oil_used = mapping["USEFO"].get(row["USEFO"])  
    solar_thermal_used = mapping["USESOLAR"].get(row["USESOLAR"])  
    wood_used = mapping["USEWOOD"].get(row["USEWOOD"])  
    all_electric = mapping["ALLELEC"].get(row["ALLELEC"])

    usage_info = f'''
When it comes to energy use, for this person, {electricity_used}, {natural_gas_used} and {propane_used}.
Moreover, for this person, {fuel_oil_used}, {solar_thermal_used} and {wood_used}. At the same time, for this person, {natural_gas_cooking} and {propane_cooking}.
    '''

    pred = row['UGASHERE']
    task = gas
    question = f'''
You are a statistician and a social survey expert. I provide you with some basic information about this person and some of his energy or fuel usage in his daily life. 
You need to accurately analyze this person's energy habits and predict whether {task} based on the information I provide.
    1. Yes
    0. No
Your response should consist of just the option (1/0) to reflect the person's opinion, without any additional text, explanation or even a space letter. 
You are not allowed to explain anything of your response. Your entire output should not exceed 1 character.
    If your answer is Yes, your response should just be 1, without any additional text or explanation.
    If your answer is No, your response should just be 0, without any additional text or explanation.    

'''

    pred_n = row['KWH']
    #'Total electricity use, in kilowatthours, 2020, including self-generation of solar power, ranging [42.01-184101.84]'
    task_numerical = mapping['Tasks'].get("nKWH")

    numerical = f'''
You are a statistician and a social survey expert. I provide you with some basic information about this person and some of his energy or fuel usage in his daily life. 
You need to accurately analyze this person's energy habits and predict the {task_numerical} of this individual based on the information I provide. 
Besides, you are not allowed to response more than a numerical number. '''
    
    return usage_info, numerical, pred_n,row['DOEID']

few_examples = '''
Here are some examples and answers:
    1. This person, who is a 65-year-old female, whose employment status is Retired and lives in the state New Mexico.
        Besides, this person's race is White Alone and the number of household members is 2, while the number of weekdays someone is at home most or all of the day is None.
        Moreover, the total household income is $60,000 - $74,999.
        When it comes to energy use, for this person, Electricity is used, Natural gas is used and Propane is not used.
        Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is used for cooking and Propane is not used for cooking.
        The answer is 12521.48

    2. This person, who is a 79-year-old female, whose employment status is Retired and lives in the state Arkansas.
        Besides, this person's race is White Alone and the number of household members is 1, while the number of weekdays someone is at home most or all of the day is 5 days.
        Moreover, the total household income is $15,000 - $19,999.

        When it comes to energy use, for this person, Electricity is used, Natural gas is used and Propane is not used.
        Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is not used for cooking and Propane is not used for cooking.
        The answer is 5243.05
'''
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/RECS/'
    input_path = "/home/cyyuan/ACL2025/Data/RECS/recs2020_public_v7.csv"
    log_path = "/home/cyyuan/ACL2025/Data/RECS/duolun/error.txt"
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        responses1 = []
        responses2 = []
        gts = []
        for i,row in enumerate(reader):
            usage_info, question, gt ,id= generate_description(row)

            
            background, base_info = generate_baseinfo(row)
            # print(base_info+usage_info)
            # print(gt)
            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ol3(conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ol31(conversation_history))
                # response1 = extract_float_string(ask_gpt3(client, conversation_history))
                # response2 = extract_float_string(ask_gpt4(client, conversation_history))
                # print(response1)
                # print(response2)
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                with open(log_path,"a") as file:
                    file.write(f"id:{id}:errortype:{e}\n")
                continue 
            with open('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round2_zero_llama.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([response1, response2, gt])
            responses1.append(response1)
            responses2.append(response2)
            gts.append(gt)
            print(f"No:{i}")
            if i>=100:
                break
            
            
    responses1=np.array(responses1)
    responses2=np.array(responses2)
    gts = np.array(gts)
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0]!=gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses1, responses2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round2_zero.csv', data, delimiter=',', header='gpt35,gpt4,gts', comments='', fmt='%s')  
    print("complete writing data into zero.csv")


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        responses1 = []
        responses2 = []
        gts = []
        for i,row in enumerate(reader):
            usage_info, question, gt ,id= generate_description(row)
            background, base_info = generate_baseinfo(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question + few_examples
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ol3(conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ol31(conversation_history))
                # response1 = extract_float_string(ask_gpt3(client, conversation_history))
                # response2 = extract_float_string(ask_gpt4(client, conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                with open(log_path,"a") as file:
                    file.write(f"id:{id}:errortype:{e}\n")
                    
                continue 
            
            responses1.append(response1)
            responses2.append(response2)
            gts.append(gt)
            with open('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round2_few_llama.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([response1, response2, gt])
            print(f"No:{i}")
            if i>=100:
                break
            
    responses1=np.array(responses1)
    responses2=np.array(responses2)
    gts = np.array(gts)
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0]!=gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses1, responses2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round2_few.csv', data, delimiter=',', header='gt35,gpt4,gts', comments='', fmt='%s')  
    print("complete writing data into few.csv")


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        responses1 = []
        responses2 = []
        gts = []
        for i,row in enumerate(reader):
            usage_info, question, gt ,id= generate_description(row)

            
            background, base_info = generate_baseinfo(row)
            # print(base_info+usage_info)
            # print(gt)
            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ol3(conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ol31(conversation_history))
                # response1 = extract_float_string(ask_gpt3(client, conversation_history))
                # response2 = extract_float_string(ask_gpt4(client, conversation_history))
                # response1 = extract_float_string(ask_gpt3(client, conversation_history))
                # response2 = extract_float_string(ask_gpt4(client, conversation_history))
                # print(response1)
                # print(response2)
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                with open(log_path,"a") as file:
                    file.write(f"id:{id}:errortype:{e}\n")
                continue 
            
            responses1.append(response1)
            responses2.append(response2)
            gts.append(gt)
            with open('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round3_zero_llama.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([response1, response2, gt])
            print(f"No:{i}")
            if i>=100:
                break
            
            
    responses1=np.array(responses1)
    responses2=np.array(responses2)
    gts = np.array(gts)
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0]!=gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses1, responses2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round3_zero.csv', data, delimiter=',', header='gpt35,gpt4,gts', comments='', fmt='%s')  
    print("complete writing data into zero.csv")


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        responses1 = []
        responses2 = []
        gts = []
        for i,row in enumerate(reader):
            usage_info, question, gt ,id= generate_description(row)
            background, base_info = generate_baseinfo(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question + few_examples
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ol3(conversation_history))  # 尝试将返回值转换为 float
                response2 = extract_float_string(ol31(conversation_history))
                # response1 = extract_float_string(ask_gpt3(client, conversation_history))
                # response2 = extract_float_string(ask_gpt4(client, conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                with open(log_path,"a") as file:
                    file.write(f"id:{id}:errortype:{e}\n")
                    
                continue 
            
            responses1.append(response1)
            responses2.append(response2)
            gts.append(gt)
            with open('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round3_few_llama.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([response1, response2, gt])
            print(f"No:{i}")
            if i>=100:
                break
            
    responses1=np.array(responses1)
    responses2=np.array(responses2)
    gts = np.array(gts)
    if responses1.shape[0] != gts.shape[0] or responses2.shape[0]!=gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses1, responses2, gts))  
    np.savetxt('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round3_few.csv', data, delimiter=',', header='gt35,gpt4,gts', comments='', fmt='%s')  
    print("complete writing data into few.csv")
if __name__ == "__main__":
    main()