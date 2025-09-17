# 生成提示词，根据回答者的基础信息，进行角色扮演，并给出预测
import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from utils import load_json

# 定义一个函数，根据条件转换值
def convert_value(val):
    try:
        float_val = float(val)
        # 如果浮点数是整数，则转换为整数
        if float_val.is_integer():
            return int(float_val)
        else:
            return float_val
    except ValueError:
        return val
    
def gen_prompt_from_fields(value, mappings, info_fields, option_field, number_field):
    # 为每一个特征field生成一句话info_desc
    info_prompt = ""
    # print(info_fields)
    # print(value)
    # print(option_field)
    # print(number_field)
    for field in info_fields:
        info_desc = ""
        fieldvalue = str(value[field])
        # print(field, fieldvalue)
        # 选择题
        if field in option_field: 
            info_desc = mappings[field].get(fieldvalue)
        # 数值题
        elif field in number_field: 
            info_desc = mappings[field] + fieldvalue + "."
        else:
            print("ERROR 1")
            print(field, fieldvalue)
        info_prompt += info_desc + " "
        # break
    return info_prompt

# Function to map values
def map_value(mappings, category, value):
    if category in mappings and value in mappings[category]:
        # print(mappings[category][value])
        return mappings[category][value]
    return str(value)

def generate_person_prompt(hh_person_df, mappings, houseid, person_info_fields, option_field, number_field):
    person_info_prompt = "The information for each household member is as follows: \n"
    # print(houseid)
    # print(hh_person_df)
    for _, person_row in hh_person_df.iterrows():
        # print(person_row)
        personid = person_row['PERSONID']
        # print(personid)
        # print(person_df)
        one_person_df = hh_person_df[hh_person_df['PERSONID'] == personid]
        one_person_info_prompt = "PersonID {}: ".format(personid)
        # print(one_person_df)    # 一个人就是一行数据
        
        value = {}
        for field in person_info_fields:
            value[field] = one_person_df.iloc[0][field] # 只有一行，只需要取第一行
        # print("++++",value)
        # PersonID = member["PersonID"]
        # age = member["R_AGE"]
        # sex = map_value("R_SEX_IMP", member["R_SEX_IMP"])
        # race = map_value("R_RACE", member["R_RACE"])
        # education = map_value("EDUC", member["EDUC"])
        # driver_status = map_value("DRIVER", member["DRIVER"])
        # worker_status = map_value("WORKER", member["WORKER"])

        # person_info_prompt += one_person_info_prompt + gen_prompt_from_fields(value, mappings, person_info_fields, option_field, number_field) + "\n"
        break
    return person_info_prompt

# 生成一个家庭的背景信息
def generate_hh_prompt(hh_cleaned_csv, person_cleaned_csv, hh_info_fields, person_info_fields, hh_prompt_csv, option_field, number_field):
    hh_df = pd.read_csv(hh_cleaned_csv)
    person_df = pd.read_csv(person_cleaned_csv)
    

    with open(hh_prompt_csv, mode='w', newline='') as file:
        # 写入表头
        writer = csv.writer(file)
        writer.writerow(["HOUSEID", "PROMPT"])
        # print(len(hh_df))
        
        for _, hh_row in tqdm(hh_df.iterrows(), total=len(hh_df), desc="Generating and saving household prompts"):    # 每一行就是每一个家庭
            # 读取数据
            value = {}
            for field in hh_info_fields:
                value[field] = str(convert_value(hh_row[field]))
            # print(value)
            houseid = int(value['HOUSEID'])
            # hhsize = int(value['HHSIZE']) 注意不能用hhsize作为迭代器来遍历所有家庭成员，因为不是所有成员都参与了问卷

            # print(houseid)
            
            # Load config
            config = load_json('config/config.json')
            # Mapping of option-prompt
            with open(os.path.join(config['mappings_folder'], 'mapping.json'), 'r') as f:
                mappings = json.load(f)

            # 生成这个家庭的基础信息prompt
            hh_fields_to_gen = hh_info_fields.copy()
            hh_fields_to_gen.remove('HOUSEID')
            # print(hh_fields_to_gen)
            # hh_info_prompt = f"Here's the basic information about one household: "
            # hh_info_prompt += gen_prompt_from_fields(value, mappings, hh_fields_to_gen, option_field, number_field) + "\n"
            # print(value["HHFAMINC"])
            hh_income = map_value(mappings, "HHFAMINC", value["HHFAMINC"])
            home_own = map_value(mappings, "HOMEOWN", value["HOMEOWN"])
            urban = map_value(mappings, "URBAN", value["URBAN"])
            hh_cbsa = map_value(mappings, "HH_CBSA", value["HH_CBSA"])
            h_size = value["HHSIZE"]
            h_veh_cnt = value["HHVEHCNT"]
            
            description = f"In a {h_size}-person family with an income of {hh_income}, who {home_own} and has {h_veh_cnt} cars, "
            description += f"living {urban} in the region of {hh_cbsa}, the information of each household member is as follows: \n"
            

            # break
            # 家庭信息生成完毕

            # 生成这个家庭中的每个人的信息
            # print(person_df)
            hh_person_df = person_df[person_df['HOUSEID'] == houseid]
            # print(person_df)
            person_fields_to_gen = person_info_fields.copy()
            person_fields_to_gen.remove('HOUSEID')
            person_fields_to_gen.remove('PERSONID')
            # print(person_fields_to_gen)
            # person_info_prompt = generate_person_prompt(hh_person_df, mappings, houseid, person_fields_to_gen, option_field, number_field)
            
            member_descriptions = []
            for _, person_row in hh_person_df.iterrows():
                # print(person_row)
                personid = person_row['PERSONID']
                # print(personid)
                # print(person_df)
                one_person_df = hh_person_df[hh_person_df['PERSONID'] == personid]
                # one_person_info_prompt = "PersonID {}: ".format(personid)
                # # print(one_person_df)    # 一个人就是一行数据
                
                value = {}
                for field in person_info_fields:
                    value[field] = one_person_df.iloc[0][field] # 只有一行，只需要取第一行
                # print(value)

                age = value["R_AGE"]
                sex = map_value(mappings, "R_SEX_IMP", str(value["R_SEX_IMP"]))
                race = map_value(mappings, "R_RACE", str(value["R_RACE"]))
                education = map_value(mappings, "EDUC", str(value["EDUC"]))
                driver_status = map_value(mappings, "DRIVER", str(value["DRIVER"]))
                worker_status = map_value(mappings, "WORKER", str(value["WORKER"]))

                member_description = f"Person {personid}: {race}, {age} years old {sex} person{education}, who {driver_status} and {worker_status}.\n"
                member_descriptions.append(member_description)
                
            description += " ".join(member_descriptions)

            # hh_info_prompt += person_info_prompt
            # print(hh_info_prompt)   # 家庭信息和每个家庭成员的信息生成完毕
            
            # 保存提示词到csv中，HOUSEID   PROMPT
            # 写入数据
            
            writer.writerow([houseid, description])
            # print(description)
            # break
    print(f"Prompts saved to {hh_prompt_csv}.\n")
    return

# 生成条件信息和问题
def generate_condq_prompt(value, mappings, cond_fields, personid, option_field, number_field):
    cond_info_prompt = f"As for the household member \\'Person {personid}\\', given the following conditions of this person's trips: "
    cond_info_prompt += gen_prompt_from_fields(value, mappings, cond_fields, option_field, number_field)
#     question_prompt = "All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day, please try your best to predict the total trip distance in miles ranging from 0 to 9621.053 and the total trip duration in minutes ranging from 0 to 1200. "
#     question_prompt = "All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day, please try your best to predict the total trip distance in miles and the total trip duration in minutes. "
#     question_prompt = """All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day,please try your best to predict the total trip distance in miles and the total trip duration in minutes, and make a choice for which kinds of trip mode they will use. The selections are follows:
# "A": "The trip mode is walking.",
# "B": "The trip mode is by private vehicle.",
# "C": "The trip mode is by bicycle, motorcycle, or moped.",
# "D": "The trip mode is by public bus.",
# "E": "The trip mode is by intercity bus, Amtrak, or commuter rail.",
# "F": "The trip mode is by subway, elevated train, light rail, or streetcar.",
# "G": "The trip mode is by taxi or rental car.",
# "H": "The trip mode is by airplane.",
# "I": "The trip mode is by boat, ferry, or water taxi."
# "J":"The trip mode is by other.",
# Here are some examples.
# Example one: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 4 cars, living in a nonurban area in the region of Minneapolis-St, Paul-Bloomington, MN-WI, the information of each household member is as follows:
# Person 1: A White, 55 years old male person with graduate degree or professional degree, who is a driver and is a worker.
#  Person 2: A White, 49 years old female person with bachelor's degree, who is a driver and is a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 225.9. The travel day is Thursday. The date of the travel day is 201608. The answers are 16.034,26,B
# Example two: In a 1-person family with an income of $125,000 to $149,999, who owns a home and has 2 cars, living in an urban area in the region of San Diego-Carlsbad, CA, the information of each household member is as follows:
# Person 1: A person whose race is classified as \'other\', 30 years old male person with bachelor's degree, who is a driver and is a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 259.6. The travel day is Sunday. The date of the travel day is 201608 The answers are 2.326, 53,A
# Example three: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 3 cars, living in an urban area in the region of Dallas-Fort Worth-Arlington, TX, the information of each household member is as follows:
# Person 1: A White, 61 years old male person with some college or associates degree, who is a driver and is not a worker.
#  Person 2: A person who has multiple races, 59 years old female person with some college or associates degree, who is a driver and is not a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 202.1. The travel day is Friday. The date of the travel day is 201612. The answers are 68.918,101,B
# Example Four:In a 2-person family with an income of $35,000 to $49,999, who owns a home and has 1 cars, living in an urban area in the region of New York-Newark-Jersey City, NY-NJ-PA, the information of each household member is as follows:
# Person 1: A White, 45 years old female person with graduate degree or professional degree, who is a driver and is a worker.
#  Person 2: A White, 45 years old male person with bachelor's degree, who is a driver and is a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 240.8. The travel day is Sunday. The date of the travel day is 201605. The answer are 11.229,77,D
#
# Example five: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 3 cars, living in an urban area in the region of Dallas-Fort Worth-Arlington, TX, the information of each household member is as follows:
# Person 1: A White, 61 years old male person with some college or associates degree, who is a driver and is not a worker.
#  Person 2: A person who has multiple races, 59 years old female person with some college or associates degree, who is a driver and is not a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 202.1. The answers are 6.524,15,J
#
# """
#     question_prompt = """All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day,please try your best to predict the total trip distance in miles and the total trip duration in minutes, and make a choice for which kinds of trip mode they will use. The selections are follows:
#     "A": "The trip mode is walking.",
#     "B": "The trip mode is by private vehicle.",
#     "C": "The trip mode is by bicycle, motorcycle, or moped.",
#     "D": "The trip mode is by public bus.",
#     "E": "The trip mode is by intercity bus, Amtrak, or commuter rail.",
#     "F": "The trip mode is by subway, elevated train, light rail, or streetcar.",
#     "G": "The trip mode is by taxi or rental car.",
#     "H": "The trip mode is by airplane.",
#     "I": "The trip mode is by boat, ferry, or water taxi."
#     "J":"The trip mode is by other.",
#
#
#     """
#     question_prompt = """All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day, please try your best to predict the total trip distance in miles and the total trip duration in minutes, and make three choices for predict the data. The selections are follows:
# For trip distance selection From A to C:
# "A": "medium distance.",
# "B": "short distance.",
# "C": "long distance.",
#
# For trip duration selection From A to C:
# "A": "Medium time trip.",
# "B": "Long time trip.",
# "C": "Short time trip.",
#
# For trip model selection From A to J:
# "A": "The trip mode is walking.",
# "B": "The trip mode is by private vehicle.",
# "C": "The trip mode is by bicycle, motorcycle, or moped.",
# "D": "The trip mode is by public bus.",
# "E": "The trip mode is by intercity bus, Amtrak, or commuter rail.",
# "F": "The trip mode is by subway, elevated train, light rail, or streetcar.",
# "G": "The trip mode is by taxi or rental car.",
# "H": "The trip mode is by airplane.",
# "I": "The trip mode is by boat, ferry, or water taxi."
# "J": "The trip mode is other."
# el day is 202.1. The travel day is Friday. The date of the travel day is 201612.The answers are B,C,J

#     """
#     fewshot = """
#     Here are some examples.
# Example one: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 4 cars, living in a nonurban area in the region of Minneapolis-St, Paul-Bloomington, MN-WI, the information of each household member is as follows:
# Person 1: A White, 55 years old male person with graduate degree or professional degree, who is a driver and is a worker.
#  Person 2: A White, 49 years old female person with bachelor's degree, who is a driver and is a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 225.9. The travel day is Thursday. The date of the travel day is 201608. The answers are B,C,B
# Example two: In a 1-person family with an income of $125,000 to $149,999, who owns a home and has 2 cars, living in an urban area in the region of San Diego-Carlsbad, CA, the information of each household member is as follows:
# Person 1: A person whose race is classified as \'other\', 30 years old male person with bachelor's degree, who is a driver and is a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 259.6. The travel day is Sunday. The date of the travel day is 201608 The answers are B, C,A
# Example three: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 3 cars, living in an urban area in the region of Dallas-Fort Worth-Arlington, TX, the information of each household member is as follows:
# Person 1: A White, 61 years old male person with some college or associates degree, who is a driver and is not a worker.
#  Person 2: A person who has multiple races, 59 years old female person with some college or associates degree, who is a driver and is not a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 202.1. The travel day is Friday. The date of the travel day is 201612. The answers are A,A,B
# Example Four: In a 2-person family with an income of $35,000 to $49,999, who owns a home and has 1 cars, living in an urban area in the region of New York-Newark-Jersey City, NY-NJ-PA, the information of each household member is as follows:
# Person 1: A White, 45 years old female person with graduate degree or professional degree, who is a driver and is a worker.
#  Person 2: A White, 45 years old male person with bachelor's degree, who is a driver and is a worker.
# As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 240.8. The travel day is Sunday. The date of the travel day is 201605. The answer are B,A,D
# Your response should consist of just two numbers and a letter separated by two commas without any additional text or explanation.
# Example five: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 3 cars, living in an urban area in the region of Dallas-Fort Worth-Arlington, TX, the information of each household member is as follows:
#     Person 1: A White, 61 years old male person with some college or associates degree, who is a driver and is not a worker.
#      Person 2: A person who has multiple races, 59 years old female person with some college or associates degree, who is a driver and is not a worker.
#     As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the trav"""
    question_prompt = """All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day,please try your best to predict the total trip distance in miles and the total trip duration in minutes, and make a choice for which kinds of trip mode they will use. The selections are follows:
    "A": "The trip mode is walking.",
    "B": "The trip mode is by private vehicle.",
    "C": "The trip mode is by bicycle, motorcycle, or moped.",
    "D": "The trip mode is by public bus.",
    "E": "The trip mode is by intercity bus, Amtrak, or commuter rail.",
    "F": "The trip mode is by subway, elevated train, light rail, or streetcar.",
    "G": "The trip mode is by taxi or rental car.",
    "H": "The trip mode is by airplane.",
    "I": "The trip mode is by boat, ferry, or water taxi."
    "J":"The trip mode is by other.",
    """
    fewshot ="""
    Here are some examples.
    Example one: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 4 cars, living in a nonurban area in the region of Minneapolis-St, Paul-Bloomington, MN-WI, the information of each household member is as follows:
    Person 1: A White, 55 years old male person with graduate degree or professional degree, who is a driver and is a worker.
     Person 2: A White, 49 years old female person with bachelor's degree, who is a driver and is a worker.
    As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 225.9. The travel day is Thursday. The date of the travel day is 201608. The answers are 16.034,25,B
    Example two: In a 2-person family with an income of $50,000 to $74,999, who owns a home and has 2 cars, living in an urban area in the region of Sacramento--Roseville--Arden-Arcade, CA, the information of each household member is as follows:
    Person 1: A White, 69 years old female person with bachelor's degree, who is a driver and is not a worker.
     Person 2: A White, 68 years old male person with bachelor's degree, who is not a driver and is not a worker.
    As for the household member \'Person 2\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 273.4. The travel day is Sunday. The date of the travel day is 201607.The answers are 18.159,50,B
    Example three: In a 2-person family with an income of $100,000 to $124,999, who owns a home and has 3 cars, living in an urban area in the region of Dallas-Fort Worth-Arlington, TX, the information of each household member is as follows:
    Person 1: A White, 61 years old male person with some college or associates degree, who is a driver and is not a worker.
     Person 2: A person who has multiple races, 59 years old female person with some college or associates degree, who is a driver and is not a worker.
    As for the household member \'Person 1\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 202.1. The travel day is Friday. The date of the travel day is 201612. The answers are 68.918,100,B
    Example Four:In a 3-person family with an income of $200,000 or more, who owns a home and has 3 cars, living in an urban area in the region of Houston-The Woodlands-Sugar Land, TX, the information of each household member is as follows:
Person 1: A White, 56 years old female person with graduate degree or professional degree, who is a driver and is a worker.
 Person 2: A White, 56 years old male person with bachelor's degree, who is a driver and is a worker.
 Person 3: A White, 17 years old female person, who is less than a high school graduate, who is a driver and is not a worker.
As for the household member \'Person 2\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 207.9. The travel day is Wednesday. The date of the travel day is 201609. The answer are 64.248,99,B

    Example five: In a 2-person family with an income of $15,000 to $24,999, who owns a home and has 2 cars, living in an urban area in the region of Milwaukee-Waukesha-West Allis, WI, the information of each household member is as follows:
Person 1: A White, 72 years old male person with bachelor's degree, who is a driver and is not a worker.
 Person 2: A White, 54 years old female person with some college or associates degree, who is a driver and is not a worker.
As for the household member \'Person 2\', given the following conditions of this person's trips: The price of the gasoline (in cents) on the travel day is 242.5. The travel day is Tuesday. The date of the travel day is 201605. The answers are 26.896,65,B

    """
    prompt_format = "Your response should consist of just two numbers and one letter separated by two comma, without any additional text or explanation." # + "\n"
    # "total trip distance in miles and the total trip duration in minutes,with the letter between A to I indicating the predicted mode of travel choice"
    question_prompt += fewshot
    # prompt_format = "Your response should consist of just three letter separated by two commas without any additional text or explanation. First answer from A-C,second answer from A-C,thrid answer from A-J"
    return cond_info_prompt + question_prompt + prompt_format #+ "\n"



def main():
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    data_root_folder = config["data_root_folder"]
    prompt_csv_folder = os.path.join(data_root_folder, 'prompt_csv')   

    os.makedirs(prompt_csv_folder, exist_ok=True)

    hh_cleaned_csv = os.path.join(data_root_folder, config["hh_csv"])
    person_cleaned_csv = os.path.join(data_root_folder, config["person_csv"])
    hh_prompt_csv = os.path.join(data_root_folder, config["hh_prompt_csv"])
    
    generate_hh_prompt(
        hh_cleaned_csv,
        person_cleaned_csv,
        config["household_cols_to_reserve"],
        config["person_cols_to_reserve"],
        hh_prompt_csv,
        set(config["option_field"]),
        set(config["number_field"])
    )

if __name__ == "__main__":
    main()
