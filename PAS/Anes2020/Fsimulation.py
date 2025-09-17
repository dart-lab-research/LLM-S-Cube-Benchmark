import pandas as pd
import csv
import ollama
def ol(messages):
    stream = ollama.chat(
        model='llama3.1',
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

    sample_info = []
    sample_info.append([tier, gender, age_group, completion, timeSpent, duration])
    return sample_info
def expert_prompt():
#     I will tell you in advance the distribution of the scores of all participants on some politicians, which can be used as your reference:
# Score distribution of Nancy Pelosi:  
#     [0-20: 0.411, 20-40: 0.082, 40-60: 0.161, 60-80: 0.152, 80-100: 0.135]  
# Score distribution of Anthony Fauci:  
#     [0-20: 0.165, 20-40: 0.067, 40-60: 0.206, 60-80: 0.177, 80-100: 0.259]  
# Score distribution of Marco Rubio:  
#     [0-20: 0.358, 20-40: 0.143, 40-60: 0.269, 60-80: 0.137, 80-100: 0.071]
    return 
def main():
    folder_path = './Data/Anes2020/'
    input_path = folder_path + 'selected_anes2020.csv'
    output_path = folder_path + 'a20_descriptions_fsimulation_Given_31.txt'
    introduction_1 = '''
You are a statistician and a social survey expert. Your task is to generate a survey dataset. 
The survey content was shaped to a significant degree by ideas offered by the ANES user community through public solicitations and by members of the ANES national advisory board. Data collection was conducted between April 10, 2020 and April 18, 2020. 
Sample was provided by three separate opt-in internet panel vendors. ANES staff programmed the survey and collected the data directly using the Qualtrics survey platform. The combined final sample includes responses from 3,080 adult citizens from across the United States.
The sample batch size is 20. The dataset should include 3 demographic variables for a respondent, whose attitude towards rating:"
'''
    introduction_2 = '''
1. How would you rate Donald Trump? [0,100]
2. How would you rate Barack Obama? [0,100]
3. How would you rate Joe Biden? [0,100]


    '''
    introduction_3 = ''' Here are some examples of data record list: 
['80', '75', '65']  
['85', '15', '20']  
['80', '50', '55']  
['0', '100', '90']  
['35', '5', '10']  
['30', '90', '40']
    Choose your answers only from the options provided. After generating, only show the data you generated without additional words. 
    The format must be a JSON string representing a three-dimensional array. Also, make sure that it is an array of arrays with no objects, like in a spreadsheet. 
    Remember, the records should closely reflect the ANES dataset."
    '''
    df = pd.read_csv(input_path) 
    with open(input_path, mode='r', newline='') as infile,open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        #writer = csv.writer(outfile)
        prompt = introduction_1+introduction_2+introduction_3
        print("prompt:", prompt)
        for i, row in enumerate(reader):
            

            # outfile.write(prompt+'\n')
            messages = [{"role": "user", "content": prompt}]
            print(f"No:{i} generating...")
            aa = ol(messages)
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            if i >= 20:
                break

if __name__ == "__main__":
    main()