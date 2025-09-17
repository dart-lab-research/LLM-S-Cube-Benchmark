import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from prompt_a20 import pp, load_mapping, base_info
      
def ol(messages):
    stream = ollama.chat(
        model='llama3.3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans


def part_info(row):
    #enrollment_features = ['meeting','moneyorg','protest','online', 'persuade','button']
    mapping = load_mapping('./Anes2020/mappings/mapping.json')

    meeting = pp(row['meeting']) + "attended a meeting to talk about political or social concerns. "
    moneyorg = pp(row['moneyorg']) + "given money to an organization concerned with a political or social issue. "
    protest = pp(row['protest']) + "joined in a protest march, rally, or demonstration. "
    online = pp(row['online']) + "posted a message or comment online about a political issue or campaign. "
    persuade = pp(row['persuade']) + "tried to persuade anyone to vote one way or another. "
    button = pp(row['button'])+ "worn a campaign button put a sticker on his/her car or placed a sign in window or in front of his/her house. "


    #pred
    pred = "tried to persuade anyone to vote one way or another "
    gt = row['persuade']


    part_description = f'''
This person's participations on political activities are as follows: 
    {button}
    {online}
    {moneyorg}
    {protest}
    {meeting}
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
    input_path = folder_path + 'selected_anes2020.csv'
    output_path = folder_path + 'A20par_descriptions.txt'
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
            part_description, question, gt = part_info(row)

            prompt = base_description + part_description + question
            conversation_history = [{"role": "user", "content": prompt}]
            response = float(ol(conversation_history))
            gt = float(gt)

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

    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + 'responses/par_persuade3.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into p3.csv")
            
        
if __name__ == "__main__":
    main()