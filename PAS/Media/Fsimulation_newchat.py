
from prompt_generate import generate_description,generate_usageinfo
from test_prompt import ol
import pandas as pd
import csv
import ollama
from openai import OpenAI
apik = "YOUR_API_KEY"  
Mapping = "D://School//UM//LLM//RECS//mappings//mapping.json"
# 调用 GPT-3.5 模型  
client = OpenAI(api_key=apik)
def ask_gpt(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
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
    #print(ans)
    # ollama_session.send({'role': 'user', 'content': introduction+prompt+question})
    # response = ''
    # for chunk in ollama_session:
    #     if 'message' in chunk and 'content' in chunk['message']:
    #         content = chunk['message']['content']
    #         print(content, end='', flush=True)
    #         response += content
    return ans
def get_sample_inform(row):
    #basic info
    tier = row['tier']
    gender = row['gender']
    age_group = row['age_group']

    #usage rate,time spent(second)
    completion = row['avgCompletion']
    timeSpent = row['avgTimeSpent']
    duration = row['avgDuration']

    sample_info = []
    sample_info.append([tier, gender, age_group, completion, timeSpent, duration])
    return sample_info
def expert_prompt():

    return 
def main():
    folder_path = './Data/Trell social media usage/'
    input_path = folder_path + 'selected_features.csv'
    output_path = r"/home/cyyuan/Data/Trell social media usage/30few.txt"
    introduction_1 = '''
You are a statistician and a social survey expert. Your task is to generate a survey dataset representing the usage of Trell, an Indian social media of videos. 
The sample size is 20. The dataset should include 6 demographic variables for a respondent:"
    
'''
    introduction_2 = '''
1. Total number of videos watched. The data you generate should be standardized data, distributed as a floating point between [0,2], with lower values indicating fewer indicators and vice versa indicating more indicators.
2. Number of videos watched on weekends per day. The data you generate should be standardized data, distributed as a floating point between [0,2], with lower values indicating fewer indicators and vice versa indicating more indicators.
3. Number of videos watched on weekdays per day. The data you generate should be standardized data, distributed as a floating point between [0,2], with lower values indicating fewer indicators and vice versa indicating more indicators.

gender Distribution in 45w individuals:
    Female: 93976 (20.93%)
    Male: 355111 (79.07%)
tier Distribution in 45w individuals:
    First-tier City: 42775 (9.52%)
    Second-tier City: 370134 (82.42%)
    Third-tier City: 36178 (8.06%)
age_group Distribution in 45w individuals:
    Less than 18: 287479 (64.01%)
    18-24: 52645 (11.72%)
    24-30: 54305 (12.09%)
    More than 30: 54658 (12.17%)
    '''
    introduction_3 = ''' Here are some examples of data record list: 
['0.2', '0.041666667', '0.025'],
['0.002785515', '1.0', '1.0557103'],
['0.2', '1.041666667', '0.025'],
['0.002785515', '0.0', '0.000557103'],
    Choose your answers only from the options provided, and please keep your letter case consistent with the example. After generating, only show the data you generated without additional words. The format must be a JSON string representing a multi-dimensional array. Also, make sure that it is an array of arrays with no objects, like in a spreadsheet. Remember, the records should closely reflect the Trell social media usage dataset."
    '''
    df = pd.read_csv(input_path) 
    with open(input_path, mode='r', newline='') as infile,open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        #writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            prompt = introduction_1+introduction_2+introduction_3

            # outfile.write(prompt+'\n')
            messages = [{"role": "user", "content": prompt}]
            aa = ol(messages)
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            print("generating",i)
            if i >= 20:
                break

if __name__ == "__main__":
    main()