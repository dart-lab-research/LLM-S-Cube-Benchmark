import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from biden_round2few import pp, load_mapping, base_info
      
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
def part_info(row):
    #enrollment_features = ['meeting','moneyorg','protest','online', 'persuade','button']
    mapping = load_mapping('/home/cyyuan/ACL2025/Anes2020/mappings/mapping.json')

    meeting = pp(row['meeting']) + "attended a meeting to talk about political or social concerns. "
    moneyorg = pp(row['moneyorg']) + "given money to an organization concerned with a political or social issue. "
    protest = pp(row['protest']) + "joined in a protest march, rally, or demonstration. "
    online = pp(row['online']) + "posted a message or comment online about a political issue or campaign. "
    persuade = pp(row['persuade']) + "tried to persuade anyone to vote one way or another. "
    button = pp(row['button'])+ "worn a campaign button put a sticker on his/her car or placed a sign in window or in front of his/her house. "


    #pred
    pred = "given money to an organization concerned with a political or social issue "
    gt = row['moneyorg']


    part_description = f'''
This person's participations on political activities are as follows: 
    {button}
    {online}
    {meeting}
    {protest}
    {persuade}

'''
    
    question = f'''According to the information, can you speculate whether this person has {pred}?  
Your response should consist of just one number (0/1) to reflect the person's attitude, without any additional text, explanation or even a space letter.
Here is an example of a required response that you should follow: 
    if you think this person has participated in this event, you should response JUST a integer 1. (WITHOUT ANY EXPLANATION!!! or the opposite you think, you should response 0)


    '''
    return part_description, question, gt

def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Anes2020/'
    input_path = '/home/cyyuan/ACL2025/Data/Anes2020/selected_anes2020.csv'
    output_path = folder_path + 'A20par_descriptions.txt'
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
            part_description, question, gt = part_info(row)

            prompt = base_description + part_description + question
            conversation_history = [{"role": "user", "content": prompt}]
            try:
                response = float(ol3(conversation_history))  # 尝试将返回值转换为 float
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            gt = float(gt)

            responses.append(response)
            gts.append(gt)
            
            # print(f"No:{i+1}"+part_description)
            # print(f"The answer is:{gt}")
            if response == gt:
                correct = correct+1
            
    responses = []
    gts = []
    acc = correct/(i+1)
    acc = format(acc*100,".2f")
    print("total num: ", i, "\nacc: ", acc)
    responses = np.array(responses)  
    gts = np.array(gts)  

    with open('/home/cyyuan/ACL2025/Data/Anes2020/duolunC/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"moneyorg_round3_zero_ollama3:  {acc}%\n")

    print("write succesfully")



    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            part_description, question, gt = part_info(row)

            prompt = base_description + part_description + question
            conversation_history = [{"role": "user", "content": prompt}]
            try:
                response = float(ol31(conversation_history))  # 尝试将返回值转换为 float
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            gt = float(gt)

            responses.append(response)
            gts.append(gt)
            
            #print(part_description)
            #print(f"The answer is:{gt}")
            if response == gt:
                correct = correct+1
            print(f"No:{i}")

            
    acc = correct/(i+1)
    acc = format(acc*100,".2f")
    print("total num: ", i, "\nacc: ", acc)
    responses = np.array(responses)  
    gts = np.array(gts)  

    with open('/home/cyyuan/ACL2025/Data/Anes2020/duolunC/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"moneyorg_round3_zero_ollama31:  {acc}%\n")

    print("write succesfully")
if __name__ == "__main__":
    main()