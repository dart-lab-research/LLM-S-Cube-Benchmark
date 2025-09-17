import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
ollama_session = ollama.chat(
    model='llama3.2',
    messages=[],
    stream=True,
)
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
def load_mapping(file_path):  
    with open(file_path, 'r') as file:  
        return json.load(file)  

def map_race(race, mapping):  
    race_str = mapping['race'].get(str(int(race)), 'unknown')  
    return race_str  

def map_sex(sex,  mapping):  
    sex_str = mapping['sex'].get(str(int(sex)), 'unknown') 
    #print(sex_str) 
    return sex_str  

def map_martial(martial, mapping):#婚姻情况
    martial_str = mapping['martial'].get(str(int(martial)), 'unknown')
    return martial_str

# mapping.json对应关系
def map_val2lab(lab, value, mapping):
    #print(lab,value)
    label = mapping[lab].get(str(int(float(value))), 'unknown')
    return label

#政策关注度
def map_views(value):
    if value == '':
        return "shows no reply to the government's policies on "
    
    value = int(float(value))
    if value == 1:
        return "thinks the government spends too little on "
    if value == 2:
        return "thinks the government spends the right amount on "
    if value == 3:
        return "thinks the government spends too much on "

#教育程度
def map_educ(educ, mapping):
    educ_str = mapping['educ'].get(str(int(educ)), 'unknown')
    return educ_str        

def base_info(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/GSS/mappings/mapping.json')  
    # id = row['id']
    # age = row['age']
    # sibs = row['sibs']
    # race = map_val2lab('race', float(row['racecen1']), mapping)  
    # sex = map_sex(float(row['sex']), mapping)  

    #deal with empty values
    id = row.get('id', 'Unknown ID')  # Default to 'Unknown ID' if not present  
    age = row.get('age', '')  # Default to empty string if not present  
    sibs = row.get('sibs', '0')  # Default to '0' if not present  
    racecen1 = row.get('racecen1', '')  # Get racecen1 value  
    sex_value = row.get('sex', '')  # Get sex value  

    # Handling racecen1 conversion safely  
    try:  
        race = map_val2lab('race', float(racecen1), mapping) if racecen1 else 'Unknown race'  
    except ValueError:  
        race = 'Invalid race value'  

    # Handling sex conversion safely  
    try:  
        sex = map_sex(float(sex_value), mapping) if sex_value else 'Unknown sex'  
    except ValueError:  
        sex = 'Invalid sex value'  
    baseInfo_parts = []  
    if id:  
        baseInfo_parts.append(f"identified by ID: {id}")  
    if age:  
        baseInfo_parts.append(f"{age}-year-old")  
    if sex:  
        baseInfo_parts.append(sex)  
    if race:  
        baseInfo_parts.append(f"belonging to the {race} community.")  
    # Join the parts, ensuring proper sentence structure  
    if baseInfo_parts:  
        baseInfo_description = "This person, " + ", ".join(baseInfo_parts)  
    else:  
        baseInfo_description = "This person has not provided sufficient information.\n"  
    # Constructing the description  
    description_parts = []  
    if sibs:  
        description_parts.append(f"they have {sibs} siblings, which suggests a lively household.")  
    else:  
        description_parts.append("they have no siblings, suggesting a quieter household.")  

    description = "In their family, " + " ".join(description_parts) + "\n\n"  




    database_background = "The General Social Survey (GSS), conducted since 1972 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.\n"
    role_play_info = "You are a statistician and a social survey expert. I will give you some information about this person's response and attitude toward different policies in GSS2022 dataset, which requires you to accurately analyze this person's behavior and make predictions about other policy responses of this person, considering the realistic situation of US in 2022.\n\n "
    
    return database_background, role_play_info, baseInfo_description + description

def hh_info(row):
    mapping = load_mapping('/home/cyyuan/ACL2025/GSS/mappings/mapping.json')  
    #婚姻，教育，收入，工作状态，工作职务，家庭构成
    marital = map_val2lab('marital', row['marital'], mapping)
    educ = map_val2lab('educ', row['educ'], mapping)
    income = map_val2lab('income', row['income'], mapping)
    wrkstat = map_val2lab('wrkstat', row['wrkstat'], mapping)
    occ = map_val2lab('occ10', row['occ10'], mapping)
    hhtype = map_val2lab('hhtype1',row['hhtype1'], mapping)

    hh_description = "This person is " + marital + ", has an education level of " + educ + ", " \
                 "earns an income of " + income + ", is currently " + wrkstat + ", " \
                 "works as a " + occ + ", and belongs to a household type of " + hhtype + "."
    
    #prediction
    polviews = map_val2lab('polviews', row['polviews'], mapping)

   
    print(hh_description)
    # mapping['martial'].get(str(int(row['martial'])), 'unknown')
    


    return 


def task_info(row):

    #given
    environ = map_views(row['natenvir']) + "improving and protecting the environment "
    heal = map_views(row['natheal']) + "improving and protecting the nation's health "
    bigcity = map_views(row['natcity']) + "solving the problems of big cities "
    crime = map_views(row['natcrime']) + "halting the rising crime rate "
    educate = map_views(row['nateduc']) + "improving the nation's education system "
    military = map_views(row['natarms']) + "the military, armaments and defense "
    welfare = map_views(row['natfare']) + "welfare "
    sscurity = map_views(row['natsoc'])+ "social security "

    #预测变量
    predict = "halting the rising crime rate "
    pred = row['natcrime']
    #
    given_info = f'''
    When talking about social survey, this person {sscurity}, {military} and {educate}. 
    Besides, this person {environ}, {heal}, {welfare} and {bigcity}
'''
    question = f'''
Considering the political situations of the United States in 2022, can you predict this person's attitude towards the government's spending on {predict}? \n 
    1.Spend too little. 
    2.Spend the right amount. 
    3.Spend too much.  

Your response should consist of just one number to reflect the person's attitude, without any additional text, explanation or even a space letter or a dot.'''
    eg = f"For example, if you think this person thinks the spending on {predict} is too little, you should response JUST 1 without any additional text and you don't have to explain your choice.\n"
    
    return given_info + question+eg,pred
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/GSS/'
    input_path = folder_path + 'selected_GSS.csv'
    output_path = folder_path + 'GSS_descriptions.txt'
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        for i,row in enumerate(reader):
            
            database_background, role_play_info, baseInfo_description= base_info(row)
            nat_info,real = task_info(row)
            #真实值无直接跳过
            if real == '':
                continue
            cnt = cnt+1
            gt = float(real)
            
            #sys.exit(0)
            #政治话题
            #task_description = task_info(row)
            prompt = database_background + role_play_info + baseInfo_description + nat_info

            #print(prompt)
            #sys.exit(0)
            conversation_history = [{"role": "user", "content": prompt}]
            response = float(ol3(conversation_history))

            responses.append(response)
            gts.append(float(gt))

            if cnt<=10:
                print("gt ", real)
                print("predict", response)
                outfile.write(f"NO: {cnt}" + prompt + f"\ngt: {gt}")
                outfile.write("###############################################\n")
            
            print(cnt)
            
            if response == gt:
                correct = correct +1 
            
            if cnt == 999:
                break
    print("acc: ", correct/cnt)
    responses = np.array(responses)  
    gts = np.array(gts)  

    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + '/responses/Crime30few.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into 30few.csv")

    sys.exit(0)


    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        for i,row in enumerate(reader):
            
            database_background, role_play_info, baseInfo_description= base_info(row)
            nat_info,real = task_info(row)
            #真实值无直接跳过
            if real == '':
                continue
            cnt = cnt+1
            gt = float(real)
            
            #sys.exit(0)
            #政治话题
            #task_description = task_info(row)
            prompt = database_background + role_play_info + baseInfo_description + nat_info

            #print(prompt)
            #sys.exit(0)
            conversation_history = [{"role": "user", "content": prompt}]
            response = float(ol31(conversation_history))

            responses.append(response)
            gts.append(float(gt))

            if cnt<=10:
                print("gt ", real)
                print("predict", response)
                outfile.write(f"NO: {cnt}" + prompt + f"\ngt: {gt}")
                outfile.write("###############################################\n")
            
            print(cnt)
            
            if response == gt:
                correct = correct +1 
            
            if cnt == 999:
                break
    print("acc: ", correct/cnt)
    responses = np.array(responses)  
    gts = np.array(gts)  

    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + '/responses/Educ31few.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into 31few.csv")
if __name__ == "__main__":
    main()