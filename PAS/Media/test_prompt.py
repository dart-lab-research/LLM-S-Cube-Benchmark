#测试单条基本信息属性
from prompt_generate import generate_description,generate_usageinfo,map_commentNum,map_contentViews,map_creation,map_slot,map_weekdays_trails,map_weekends_trails
import csv
import pandas as pd
import ollama
import sys
import numpy as np

question_gender = '''
Can you tell me what's the gender of this person? 
    1. male
    2. female
The predictions should be presented as a concise numerical output(1/2/3), without any additional text or explanation. Here is an example output if you choose male:
    1

    You are not allowed to response more than a numerical number.
'''
question_tier = '''
Can you help me predict what's the city tier of this person? 
    1. first-tier city
    2. second-tier city
    3. third-tier city
The predictions should be presented as a concise numerical output(1/2/3), without any additional text or explanation. Here is an example output if you choose third-tier city:
    3

    You are not allowed to response more than a numerical number.
'''
def in_context(row):
    example_prompt = ""

    return example_prompt
# def generate_question(row):
#     #normalized info
#     numPerAction = map_commentNum(row['number_of_words_per_action'])
#     creation = map_creation(row['creations'])
#     conView = map_contentViews(row['content_views'])
#     WeekendsWatch = map_weekends_trails(row['weekends_trails_watched_per_day'])
#     WeekdaysWatch = map_weekdays_trails(row['weekdays_trails_watched_per_day'])
#     slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
#     slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
#     slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
#     slot4 = map_slot(row['slot4_trails_watched_per_day'],4)

#     description_info = generate_description(row)
#     description_predict = f"When talk about this person's habit in Trell usage, here are some"
ollama_session = ollama.chat(
    model='llama3.2',
    messages=[],
    stream=True,
)
def ol(messages):

    stream = ollama.chat(
        model='llama3.3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    #print(ans)
    # ollama_session.send({'role': 'user', 'content': introduction+prompt+question})
    # response = ''
    # for chunk in ollama_session:
    #     if 'message' in chunk and 'content' in chunk['message']:
    #         content = chunk['message']['content']
    #         print(content, end='', flush=True)
    #         response += content
    return ans


def gpt(introduction, prompt, question):
    answer = ""

    return answer
def main():
    folder_path = './Data/Trell social media usage/'
    input_path = folder_path + 'random_features.csv'
    output_path = folder_path + 'prompt_3.txt'

    introduction_prompt = f'''Trell is an Indian social media and content creation app that allows users to create and share video content, focusing on travel, lifestyle, and various experiences. 
    Now you are role-playing this person based on the above information.'''

    count_correct = 0
    count_incorrect = 0
    cnt = 0
    m = 0

    f = 0
    s = 0
    t = 0
    conversation_history = []
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            
            description = generate_description(row)
            usageinfo = generate_usageinfo(row)

            predict_gender = row['gender']
            predict_tier = row['tier']
            
            prompt = description + '\n' + usageinfo
            
            if cnt <= 3:
                prompt = introduction_prompt + prompt
            

            if predict_gender == "male":
                m = m+1
            else:
                f = f+1
            prompt = prompt + question_gender
            
            #print(prompt)
            #writer.write(prompt)
            conversation_history = [{"role": "user", "content": prompt}]
            response = ol(conversation_history)

            # #print(conversation_history)
            # #print("response: ",response, "true: ",predict_gender)
            # # conversation_history.append({"role": "assistant", "content": response})
            # # print(gpt_response)

            # if predict_tier == "first-tier city":
            #     f = f+1
            #     a = 1
            # elif predict_tier == "second-tier city":
            #     s = s+1
            #     a = 2
            # else:
            #     t = t+1
            #     a = 3

            if predict_gender == "male":
                a = 1
            else:
                a = 2
            response = float(response[0])

            print(f"No:{i+1}: response:{response} gt:{a}")
            responses.append(response)
            gts.append(a)
            if response == a:
                count_correct = count_correct+1
            else:
                count_incorrect =count_incorrect+1

            if i<5:
                outfile.write(f"No:{i}\n"+ prompt)
            if i==499:
                break
    
    accuracy =count_correct/(count_correct+count_incorrect)
    print(f"acc in {i+1} rounds:", accuracy)
    print("f : s : t == ",f," : ",s," : ",t)

    responses = np.array(responses)  
    gts = np.array(gts)  

    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + '/responses/Gender3.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into Gender3.csv")
if __name__ == "__main__":
    main()