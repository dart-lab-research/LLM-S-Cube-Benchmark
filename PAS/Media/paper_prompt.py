
from prompt_generate import generate_description,generate_usageinfo
import ollama
import pandas as pd
import csv
import sys
def normalized_prompt(row):
    conView = row['content_views']
    weekends = row['weekends_trails_watched_per_day']
    weekdays = row['weekdays_trails_watched_per_day']
    return 
def ol(messages):

    stream = ollama.chat(
        model='llama3.1:latest',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
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

    conView = row['content_views']
    weekends = row['weekends_trails_watched_per_day']
    weekdays = row['weekdays_trails_watched_per_day']

    sample_info = []
    sample_info.append([tier, gender, age_group, conView, weekends, weekdays])
    return sample_info
def expert_prompt():

    return 
def main():
    folder_path = './Data/Trell social media usage/'
    input_path = folder_path + 'selected_features.csv'
    output_path = folder_path+ 'paper_prompt.txt'
    introduction_1 = "You are a statistician and a social survey expert. Your task is to generate a survey dataset representing the usage of Trell, an Indian social media of videos. The sample size is 10. The dataset should include 6 demographic variables for a respondent:\n"
    introduction_2 = '''
1. Tier(first/second/third-tier city).
2. Gender.
3. Age group(less than 18 years old/ages between 18 and 24 years old/ages between 24 and 30 years old/more than 30 years old).
4. Total number of videos watched. The data you generate should be standardized data, distributed as a floating point between [0,1], with lower values indicating fewer indicators and vice versa indicating more indicators.
5. Number of videos watched on weekends per day. The data you generate should be standardized data, distributed as a floating point between [0,1], with lower values indicating fewer indicators and vice versa indicating more indicators.
6. Number of videos watched on weekdays per day. The data you generate should be standardized data, distributed as a floating point between [0,1], with lower values indicating fewer indicators and vice versa indicating more indicators.

'''

    introduction_3 = ''' 
Choose your answers only from the options provided, and please keep your letter case consistent with the example. 
After generating, only show the data you generated without additional words. The format must be a JSON string representing a multi-dimensional array. 
Also, make sure that it is an array of arrays with no objects, like in a spreadsheet. Remember, the records should closely reflect the Trell social media usage dataset.
'''
    df = pd.read_csv(input_path) 
    with open(input_path, mode='r', newline='') as infile,open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        #writer = csv.writer(outfile)
        sample_info = []
        messages = []
        prompt = introduction_1+introduction_2+introduction_3
        messages.append({"role": "user", "content": prompt})

        print(prompt)
        for i,row in enumerate(reader):
            response = ol(messages)

            #print(response)
            sample_info = get_sample_inform(row)
            prompt = f"Here is an example of a data record of No.{i} example:\n " + "\n".join([str(sample_info)])
            print(str(sample_info))
            messages = [{"role": "user", "content": prompt}]

            outfile.write(f"Question: {prompt}\n\n")  
            outfile.write(f"Model Response: {response}\n\n")  
            outfile.write("-" * 50 + "\n\n")  # 分隔线  

            #print(response)
            #多轮对话
            # 在introduction_3加入这句话：Everytime I send you an example data, you should generate a batch of 10 records. After several times of sending, the total records you response should reflect a properiate distribution.
            
            # if i<=5:
            #     sample_info.append(get_sample_inform(row))
            
            # messages.append({"role": "assistant", "content": response})
            # prompt = f"Here is an example of a data record of No.{i} example:\n " + "\n".join([str(sample) for sample in sample_info]) + "\n You should generate a batch of records and the batch size is 100. "
            # messages.append({"role": "assistant", "content": prompt})

            
            
            if i>=5:
                break

        sys.exit(0)
        outfile.write(prompt+'\n')

        messages.append({"role": "user", "content": prompt})
        response = ol(messages)
        messages.append({"role": "user", "content": response})


       
        print(f"Question: {prompt}")  
        print(f"Model Response: {response}")  
        print("-" * 50)  # 分隔线  
            

if __name__ == "__main__":
    main()