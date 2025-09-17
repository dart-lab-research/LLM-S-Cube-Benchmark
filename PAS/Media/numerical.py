#选择题
from prompt_generate import generate_description,generate_usageinfo,map_commentNum,map_creation,map_slot,map_weekdays_trails,map_weekends_trails
import csv
import pandas as pd
import ollama
import sys
import re
import numpy as np
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
    model='llama3.2',
    messages=[],
    stream=True,
)
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
def  generate_numerical(row):
    #given numPeraction,creation,weekdayswatch   predict conview,weekendswatch
    numPerAction = map_commentNum(row['number_of_words_per_action'])
    creation = map_creation(row['creations'])
    WeekdaysWatch = map_weekdays_trails(row['weekdays_trails_watched_per_day'])

    #prediction
    conView = map_score2option(row['content_views'])
    WeekendsWatch = map_score2option(row['weekends_trails_watched_per_day'])
    
    slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
    slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
    slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
    slot4 = map_slot(row['slot4_trails_watched_per_day'],4)

    given_info = f"This one {numPerAction}, while at the same time {creation}. Besides, this person {WeekdaysWatch}\n"
    question = '''According to the information, can you help me choose the video playback habits of this person? 
                Your response should consist of just the option A,B,C,D,E or F without any additional text, explanation or even a space letter(Here is an example of a required reponse that you should follow: [1.A,2.B]) 
                1.  Video watching habit:
                    A. Watch very few videos [0-0.2].
                    B. Watch fewer videos [0.2-0.4].
                    C. Watch some videos [0.4-0.6].
                    D. Watch more videos [0.6,0.8].
                    E. Watch lots of videos [0.8,1.0].
                    F. Watch massive videos [1.0,...].
                
                2.  Videos watching on weekdays (per day):
                        A. Watch very few videos [0-0.2].
                        B. Watch fewer videos [0.2-0.4].
                        C. Watch some videos [0.4-0.6].
                        D. Watch more videos [0.6,0.8].
                        E. Watch lots of videos [0.8,1.0].
                        F. Watch massive videos [1.0,...].

                '''
                #     Here are some examples and answers:
                # 1. Average duration of the videos that this person has watched till date is 93.912.This one uses more words in comments(normalized score 0.714285714, in range of [0.6,0.8]), while at the same time uploads very few videos(normalized score 0.0234375, in range of [0,0.2]). Besides, this person watches very few videos on weekdays per day(normalized score 0.0046875, in range of [0,0.2])
                # The answer is AA

                # 2. Average duration of the videos that this person has watched till date is 119.2845.This one comments with fewer words(normalized score 0.341463415, in range of [0.2,0.4]), while at the same time uploads very few videos(normalized score 0.024, in range of [0,0.2]). Besides, this person watches very few videos on weekdays per day(normalized score 0.016, in range of [0,0.2])
                # The answer is BB
                # 2.  Videos watching on weekends per day:
                #     A. Watch very few videos [0-0.2].
                #     B. Watch fewer videos [0.2-0.4].
                #     C. Watch some videos [0.4-0.6].
                #     D. Watch more videos [0.6,0.8].
                #     E. Watch lots of videos [0.8,1.0].
                #     F. Watch massive videos [1.0,...].
    return given_info, question ,conView + WeekendsWatch
def main():
    folder_path = './Data/Trell social media usage/'
    input_path = folder_path + 'random_features.csv'
    output_path = folder_path + 'responses/numerical.txt'

    introduction_prompt = "Trell is an Indian social media and content creation app that allows users to create and share video content, focusing on travel, lifestyle, and various experiences. Now you are role-playing this person based on the above information."
    
    cnt = 0
    t1 = 0
    t2=0
    responses1 = []
    gts1 = []
    responses2 = []
    gts2 = []
    response = "AA"
    df = pd.read_csv(input_path) 
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        for i,row in enumerate(reader):

            description = generate_description(row)
            given_info, question, pred = generate_numerical(row)

            prompt = introduction_prompt + description + given_info + question + '\n'

            conversation_history = [{"role": "user", "content": prompt}]
            response = ol(conversation_history)

            #print("answer: ",pred)
            #pred为答案，response为提取后答案
            #get option
            matches = re.findall(r'\d\.\s?([A-Z])', response)  
            response = ''.join(matches) 
            
            #print(pred)
            #排除不规范形式
            if len(response) != 2 or not response.isalpha() or not response.isupper(): 
                continue
            if response[0] == pred[0]:
                t1 = t1+1
            if response[1] == pred[1]:
                t2 = t2+1
            cnt = cnt+1

            # responses1.append(response[0])
            # gts1.append(pred[0])
            # responses2.append(response[1])
            # gts2.append(pred[1])
            if cnt<=10:
                print("response: ",response)
                print("answer: ",pred)
                outfile.write(prompt)
                outfile.write("response: "+ response)
                outfile.write('\n'+pred)

            print(f"No:{i+1}: response:{response} gt:{pred}")

            if i>=1000:
                break
    acc1 = t1/(i+1)   #两道题
    acc2 = t2/(i+1)
    print(f"acc1: {acc1}  acc2:{acc2}")     

    # responses1 = np.array(responses1)  
    # gts1 = np.array(gts1)  
    # responses2 = np.array(responses2)  
    # gts2 = np.array(gts2)  
    # # 确保两个数组的长度相同  
    # if responses1.shape[0] != gts1.shape[0] or responses2.shape[0] != gts2.shape[0]:  
    #     raise ValueError("responses 和 gts 数组的长度必须相同")  
    # data = np.column_stack((responses1, gts1, responses2, gts2))  
    # np.savetxt(folder_path + '/responses/few_acc31.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    # print("complete writing data into habit3.csv")  
    # return 

if __name__ == "__main__":
    main()