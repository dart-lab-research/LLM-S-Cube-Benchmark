import pandas as pd
import csv
import json
import ollama
import numpy as np
from utils import ask_gpt4, ask_gpt3, apik, client
from prompt_test import extract_float_string
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
When it comes to energy use, for this person, {gas}, {natural_gas_used} and {electricity_used}.
Moreover, for this person, {fuel_oil_used}, {solar_thermal_used} and {wood_used}. At the same time, for this person, {natural_gas_cooking} and {propane_cooking}.
    '''

    pred = row['USELP']
    task = propane_used

    # print("task:",task)
    # print(pred)
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
    

    
    return usage_info, question, pred

few_examples = '''

        Here are some examples and answers:
            1. This person, who is a 53-year-old female, whose employment status is Employed full-time and lives in the state Texas.
            Besides, this person's race is White Alone and the number of household members is 5, while the number of weekdays someone is at home most or all of the day is 5 days.
            Moreover, the total household income is $20,000 - $24,999.
            When it comes to energy use, for this person, Natural gas is available in neighborhood., Natural gas is used and Electricity is used.
            Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is used for cooking.
            The answer is 0

            2. This person, who is a 65-year-old female, whose employment status is Retired and lives in the state New Mexico.
            Besides, this person's race is White Alone and the number of household members is 2, while the number of weekdays someone is at home most or all of the day is None.
            Moreover, the total household income is $60,000 - $74,999.
            When it comes to energy use, for this person, Natural gas is available in neighborhood., Natural gas is used and Propane is not used.
            Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is used for cooking and Propane is not used for cooking.    
            The answer is 0

        The predictions should be presented as a concise numerical output. No calculation process required. Besides, when outputting numbers, please use pure numeric format and do not add commas. For example, if your answer is '10,000', you should response '10000' without the comma.
'''
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/RECS/'
    input_path = "/home/cyyuan/ACL2025/Data/RECS/selected_RECS.csv"

    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        for i,row in enumerate(reader):
            usage_info, question, gt = generate_description(row)

            
            background, base_info = generate_baseinfo(row)
            if gt == 1:
                print(base_info+usage_info)
                print(gt)
            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))
                response2 = extract_float_string(ask_gpt4(client, conversation_history))

                # response1 = extract_float_string(ol3(conversation_history))
                # response2 = extract_float_string(ol31(conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            
            if response1 == gt:
                correct1 = correct1+1
            if response2 == gt:
                correct2 = correct2+1

            print(f"res1:{response1}; res2:{response2}; gt{gt}; correct1:{correct1}; correct2:{correct2}")
            print(f"No:{i}")
            if i >= 101:
                break
    print(i)        
    cnt = i - cnt + 1
    acc1 = correct1/(i)
    acc2 = correct2/(i)
    acc1 = format(acc1*100,".2f")
    acc2 = format(acc2*100,".2f")

    with open('/home/cyyuan/ACL2025/Data/RECS/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"USELP_round2_zero_gpt35:  {acc1}%\n")
        file.write(f"USELP_round2_zero_gpt4:  {acc2}%\n")

    print("write successfully")


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        for i,row in enumerate(reader):
            usage_info, question, gt = generate_description(row)
            background, base_info = generate_baseinfo(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question + few_examples
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
                # response1 = extract_float_string(ol3(conversation_history))
                # response2 = extract_float_string(ol31(conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            
            if response1 == gt:
                correct1 = correct1+1
            if response2 == gt:
                correct2 = correct2+1
            print(f"No:{i}")
            if i >= 101:
                break

            
    cnt = i - cnt + 1
    acc1 = correct1/(i)
    acc2 = correct2/(i)
    acc1 = format(acc1*100,".2f")
    acc2 = format(acc2*100,".2f")

    with open('/home/cyyuan/ACL2025/Data/RECS/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"USELP_round2_few_gpt35:  {acc1}%\n")
        file.write(f"USELP_round2_few_gpt4:  {acc2}%\n")

    print("write successfully")

    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        for i,row in enumerate(reader):
            usage_info, question, gt = generate_description(row)

            
            background, base_info = generate_baseinfo(row)
            if gt == 1:
                print(base_info+usage_info)
                print(gt)
            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
                # response1 = extract_float_string(ol3(conversation_history))
                # response2 = extract_float_string(ol31(conversation_history))

            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            
            if response1 == gt:
                correct1 = correct1+1
            if response2 == gt:
                correct2 = correct2+1

            
            print(f"No:{i}")
            if i >= 101:
                break
            
            
    cnt = i - cnt + 1
    acc1 = correct1/(i)
    acc2 = correct2/(i)
    acc1 = format(acc1*100,".2f")
    acc2 = format(acc2*100,".2f")

    with open('/home/cyyuan/ACL2025/Data/RECS/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"USELP_round3_zero_gpt35:  {acc1}%\n")
        file.write(f"USELP_round3_zero_gpt4:  {acc2}%\n")

    print("write successfully")


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        correct1 = 0
        correct2 = 0
        for i,row in enumerate(reader):
            usage_info, question, gt = generate_description(row)
            background, base_info = generate_baseinfo(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question + few_examples
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            try:
                response1 = extract_float_string(ask_gpt3(client, conversation_history))
                response2 = extract_float_string(ask_gpt4(client, conversation_history))
                # response1 = extract_float_string(ol3(conversation_history))
                # response2 = extract_float_string(ol31(conversation_history))
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            
            if response1 == gt:
                correct1 = correct1+1
            if response2 == gt:
                correct2 = correct2+1
            print(f"No:{i}")
            if i >= 101:
                break

            
    cnt = i - cnt + 1
    acc1 = correct1/(i)
    acc2 = correct2/(i)
    acc1 = format(acc1*100,".2f")
    acc2 = format(acc2*100,".2f")

    with open('/home/cyyuan/ACL2025/Data/RECS/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"USELP_round3_few_gpt35:  {acc1}%\n")
        file.write(f"USELP_round3_few_gpt4:  {acc2}%\n")

    print("write successfully")


if __name__ == "__main__":
    main()