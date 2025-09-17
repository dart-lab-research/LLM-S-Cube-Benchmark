#选择题
from prompt_generate import generate_description,generate_usageinfo,map_commentNum,map_creation,map_slot,map_weekdays_trails,map_weekends_trails,map_contentViews
import csv
import pandas as pd
import ollama
import sys
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy 
from scipy.interpolate import make_interp_spline  
from utils import ask_gpt4, ask_gpt3, apik, client
# from plot import num_plot1

def info_generate(row):
    numPerAction = map_commentNum(row['number_of_words_per_action'])
    WeekendsWatch = map_weekends_trails(row['weekends_trails_watched_per_day'])
    
    conView = map_contentViews(row['content_views'])
    #prediction
    
    creation = map_score2option(row['creations'])
    WeekdaysWatch = map_score2option(row['weekdays_trails_watched_per_day'])

    slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
    slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
    slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
    slot4 = map_slot(row['slot4_trails_watched_per_day'],4)
    given_info = f"This one {numPerAction}, while at the same time {conView}. Besides, this person {WeekendsWatch}\n"
    question = '''According to the information, can you help me choose the video playback habits of this person on weekdays? 
                    Your response should consist of just the option A,B,C,D,E or F without any additional text, explanation or even a space letter(Here is an example of a required reponse that you should follow: A) 
                        A. Watch very few videos [0-0.2].
                        B. Watch fewer videos [0.2-0.4].
                        C. Watch some videos [0.4-0.6].
                        D. Watch more videos [0.6,0.8].
                        E. Watch lots of videos [0.8,1.0].
                        F. Watch massive videos [1.0,...].
                    '''
    return given_info, question ,map_score2option(row['weekdays_trails_watched_per_day']),row['userId']

def map_score2option(a): #turn to options
    a = float(a)
    if 0 <= a < 0.2:
        return 'A'
    elif 0.2 <= a < 0.4:
        return 'B'
    elif 0.4 <= a < 0.6:
        return 'C'
    elif 0.6 <= a < 0.8:
        return 'D'
    elif 0.8 <= a <= 1.0:
        return 'E'
    else:
        return 'F'

ollama_session = ollama.chat(
    model='llama3.1',
    messages=[],
    stream=True,
)
def safe_float_conversion(s):  
    # 使用正则表达式匹配合法的浮点数  
    match = re.search(r'[-+]?\d*\.?\d+', s)  
    if match:  
        return float(match.group())  
    else:  
        raise ValueError(f"Invalid float format: {s}")  
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

def replace_float(code):
    # 定义正则表达式，匹配ol3后面跟着返回值，并且该返回值是由大写字母A-F组成
    pattern = r'ol3\((.*?)\)\s*=\s*float\((.*?)\)'

    # 定义替换的函数
    def replacement(match):
        # 获取 ol3 函数的返回值
        ol3_result = match.group(2).strip()
        # 检查值是否是唯一一个字符，并且该字符是不是在大写字母 A-F 范围内
        if len(ol3_result) == 1 and ol3_result in 'ABCDEF':
            return f"ol3({match.group(1)}) = ol3_result({ol3_result})"
        return match.group(0)  # 如果不满足条件，返回原始匹配

    # 使用 re.sub 进行替换
    modified_code = re.sub(pattern, replacement, code)
    return modified_code
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

def  generate_numerical(row):
    #given numPeraction,creation,weekdayswatch   predict conview,weekendswatch
    numPerAction = map_commentNum(row['number_of_words_per_action'])
    creation = map_creation(row['creations'])
    WeekdaysWatch = map_weekdays_trails(row['weekdays_trails_watched_per_day'])
    conView = map_contentViews(row['content_views'])
    WeekendsWatch = map_weekends_trails(row['weekends_trails_watched_per_day'])
    #prediction
    
    task = "videos watched per day on weekdays(normalized)"
    pred = row['weekdays_trails_watched_per_day']

    slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
    slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
    slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
    slot4 = map_slot(row['slot4_trails_watched_per_day'],4)

    given_info = f"This one {numPerAction}, while at the same time {creation}. Besides, this person {conView} and {WeekendsWatch}.\n"
    # Some examples and answers are as follows:

    #             1. Average duration of the videos that this person has watched till date is 93.912. This one uses more words in comments(normalized score 0.714285714, in range of [0.6,0.8]), while at the same time uploads very few videos(normalized score 0.0234375, in range of [0,0.2]). Besides, this person watches very few videos on weekdays per day(normalized score 0.0046875, in range of [0,0.2]) and watches very few videos(normalized score 0.0234375, in range of [0,0.2]).
    #             The answer is 0.0234375

    #             2.  Average duration of the videos that this person has watched till date is 237.921.This one comments with very few words(normalized score 0.0, in range of [0,0.2]), while at the same time uploads very few videos(normalized score 0.0, in range of [0,0.2]). Besides, this person watches very few videos on weekdays per day(normalized score 0.0, in range of [0,0.2]) and watches very few videos(normalized score 0.001814882, in range of [0,0.2]).
    #             The answer is 0.001814882
    question = f'''According to the information, can you help me predict the {task} of this person?  
                Your response should consist of just one normalized number to reflect the person's habit, without any additional text, explanation or even a space letter.

                Here is an example of a required reponse that you should follow: 
                    if you think this person has weak {task}, you should response JUST a float number like 0.032. (or more you think, the number may be bigger)
                So your response should be like 0.032(Without any additional text! Just the number, NOT 0.032. but 0.032)
                '''
    
    return given_info, question ,pred,row['userId']

few_examples = '''
    Some examples and answers are as follows:
        1. This one comments with very few words(normalized score 0.0, in range of [0,0.2]), while at the same time A. Besides, this person and very few videos on weekends per day(normalized score 0.041666667, in range of [0,0.2])
        The answer is:A
        2. This one comments with very few words(normalized score 0.153846154, in range of [0,0.2]), while at the same time A. Besides, this person and very few videos on weekends per day(normalized score 0.012711864, in range of [0,0.2])
        The answer is:A
        3. This one comments with very few words(normalized score 0.0, in range of [0,0.2]), while at the same time A. Besides, this person and very few videos on weekends per day(normalized score 0.0, in range of [0,0.2])
        The answer is:A
        4. This one comments with very few words(normalized score 0.0, in range of [0,0.2]), while at the same time A. Besides, this person and very few videos on weekends per day(normalized score 0.0, in range of [0,0.2])
        The answer is:A
        5. This one comments with very few words(normalized score 0.0, in range of [0,0.2]), while at the same time A. Besides, this person and very few videos on weekends per day(normalized score 0.0, in range of [0,0.2])
        The answer is:B
'''
def main():
    #/home/cyyuan/Data/Trell social media usage/random_features.csv
    folder_path = './Data/Trell social media usage/'
    input_path = '/home/cyyuan/ACL2025/Data/Trell social media usage/train_age_dataset.csv'
    log_path = '/home/cyyuan/ACL2025/Media/num2_choose.py'
    responses1 = []
    responses2 = []
    preds = []
    introduction_prompt = "Trell is an Indian social media and content creation app that allows users to create and share video content, focusing on travel, lifestyle, and various experiences. Now you are role-playing this person based on the above information."  
    cnt = 0

    for round in range(2):
        with open(input_path, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)
            correct1 = 0
            correct2 = 0
            cnt = 0
            for i,row in enumerate(reader):

                description = generate_description(row)
                given_info, question, pred,id = info_generate(row)
                
                # print(given_info)
                # print(f"The answer is :{pred}")
                #以防0太多干扰判断
                prompt = introduction_prompt + description + given_info + question
                
                with open("/home/cyyuan/ACL2025/Media/prompt.txt","a") as prompt_file:
                    prompt_file.write(prompt+"\n")
                #print(prompt)
                conversation_history = [{"role": "user", "content": prompt}]
                try:
                    #response1 = replace_float(ol3(conversation_history))  # 尝试将返回值转换为 float
                    #response2 = replace_float(ol31(conversation_history))
                     response1 = replace_float(ask_gpt3(client, conversation_history))
                     response2 = replace_float(ask_gpt4(client, conversation_history))
                except (ValueError, TypeError) as e:
                    # 捕获转换失败的情况并跳过当前循环
                    #print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                    with open(log_path,"a") as file:
                        file.write(f"No:{id},error type:{e}\n")
                    continue 
                
                #print("pred:",pred,"reponse2",response1,response2[0])
                response1 = response1[0]
                response2 = response2[0]
                cnt=cnt+1
                if response1 == pred:
                    correct1 = correct1+1
                if response2 == pred:
                    correct2 = correct2+1
                print(f"No:{i}: {correct1}/{correct2}  answer:{pred} response{response1}")
                
                #print(f"{i+1}.",given_info,f"The answer is:{pred}")
                if i>=10:
                    break
        acc1 = correct1/cnt
        acc2 = correct2/cnt
        acc1 = format(acc1*100,".2f")
        acc2 = format(acc2*100,".2f")

        with open('/home/cyyuan/ACL2025/Data/Trell social media usage/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
            file.write(f"weekdays_round{round+1}_zero_gpt35:  {acc1}%\n")
            file.write(f"weekdays_round{round+1}_zero_gpt4:  {acc2}%\n")

        print("write successfully")

        sys.exit(0)

        with open(input_path, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)
            correct1 = 0
            correct2 = 0
            cnt = 0
            for i,row in enumerate(reader):

                description = generate_description(row)
                given_info, question, pred,id = info_generate(row)
                
                #以防0太多干扰判断
                prompt = introduction_prompt + description + given_info + question + few_examples

                #print(prompt)
                conversation_history = [{"role": "user", "content": prompt}]
                try:
                    response1 = replace_float(ol3(conversation_history))  # 尝试将返回值转换为 float
                    response2 = replace_float(ol31(conversation_history))

                    # response1 = replace_float(ask_gpt3(client, conversation_history))
                    # # response2 = replace_float(ask_gpt4(client, conversation_history))
                except (ValueError, TypeError) as e:
                    # 捕获转换失败的情况并跳过当前循环
                    #print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                    with open(log_path,"a") as file:
                        file.write(f"No:{id},error type:{e}\n")
                    continue 
                
                #print("pred:",pred,"reponse1,2",response1,response2)
                response1 = response1[0]
                response2 = response2[0]
                cnt=cnt+1
                if response1 == pred:
                    correct1 = correct1+1
                if response2 == pred:
                    correct2 = correct2+1
                print(f"{correct1}/{correct2}")
                print(f"No:{i}")
                
                print(f"{i+1}.",given_info,f"The answer is:{pred}")
                
                if i>=300:
                    break
        acc1 = correct1/cnt
        acc2 = correct2/cnt
        acc1 = format(acc1*100,".2f")
        acc2 = format(acc2*100,".2f")

        with open('/home/cyyuan/ACL2025/Data/Trell social media usage/duolun/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
            file.write(f"weekdays_round{round+1}_few_gpt35:  {acc1}%\n")
            file.write(f"weekdays_round{round+1}_few_gpt4:  {acc2}%\n")

        print("write successfully")

if __name__ == "__main__":
    main()