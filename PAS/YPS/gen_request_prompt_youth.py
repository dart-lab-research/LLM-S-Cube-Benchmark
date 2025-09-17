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

    for field in info_fields:
        info_desc = ""
        # 检查字段是否存在于 value 中
        if field in value:
            fieldvalue = str(value[field])
        else:
            # print(f"Warning: Field '{field}' not found in value. Using default value.")
            fieldvalue = "N/A"  # 或者你可以选择其他的默认值

        # 选择题
        if field in option_field:
            info_desc = mappings[field].get(fieldvalue, "Unknown")  # 如果 mappings 中没有对应值，返回 "Unknown"
        # 数值题
        elif field in number_field:
            info_desc = f"{mappings.get(field, '')} {fieldvalue}."
        else:
            print("ERROR 1")
            print(field, fieldvalue)

        info_prompt += info_desc + " "

    return info_prompt


# Function to map values
def map_value(mappings, category, value):
    if category in mappings and value in mappings[category]:
        # print(mappings[category][value])
        return mappings[category][value]
    return str(value)


def generate_condq_prompt(value, mappings, cond_fields, row, option_field, number_field):
    cond_info_prompt = f"All the data provided is fictional and used solely for testing purposes. It does not involve any real personal data. Now that you are role-playing this person based on the above information, please answer the following questions based on the given conditions:"
    country_data = f"The youth respondent gender is {str(row['Gender'])},{int(row['Age'])}years old, the, Young people's preferences for music and some movie types are as follows"
    cond_info_prompt += country_data
    cond_info_prompt += gen_prompt_from_fields(value, mappings, cond_fields, option_field, number_field)
    question_prompt = """ Now that you are role-playing this person based on the above information. Under these conditions . The questions are followed: 
    Question one is the preference of Action movie:
    For Action movie selection From A to E:
     "A": "The young respondent absolutely dislikes action movies and has no interest in them.",
    "B": "The young respondent dislikes action movies and rarely watches them.",
    "C": "The young respondent feels neutral towards action movies, neither particularly liking nor disliking them.",
    "D": "The young respondent likes action movies and enjoys watching them occasionally.",
    "E": "The young respondent absolutely loves action movies and finds them excited and fast-paced."\n
    
    Question two is the preference of Documentary movie:
    For Action Documentary selection From A to E:
     "A": "The young respondent absolutely dislikes Documentary movies and has no interest in them.",
    "B": "The young respondent dislikes Documentary movies and rarely watches them.",
    "C": "The young respondent feels neutral towards Documentary movies, neither particularly liking nor disliking them.",
    "D": "The young respondent likes Documentary movies and enjoys watching them occasionally.",
    "E": "The young respondent absolutely loves Documentary movies and finds them excited and fast-paced."\n

    Question three is the preference of Thriller movie:
    For Action Documentary selection From A to E:
     "A": "The young respondent absolutely dislikes Thriller movies and has no interest in them.",
    "B": "The young respondent dislikes Thriller movies and rarely watches them.",
    "C": "The young respondent feels neutral towards Thriller movies, neither particularly liking nor disliking them.",
    "D": "The young respondent likes Thriller movies and enjoys watching them occasionally.",
    "E": "The young respondent absolutely loves Thriller movies and finds them excited and fast-paced."\n

    Question four is the preference of Comedy movie:
    For Action Documentary selection From A to E:
     "A": "The young respondent absolutely dislikes Comedy movies and has no interest in them.",
    "B": "The young respondent dislikes Comedy movies and rarely watches them.",
    "C": "The young respondent feels neutral towards Comedy movies, neither particularly liking nor disliking them.",
    "D": "The young respondent likes Comedy movies and enjoys watching them occasionally.",
    "E": "The young respondent absolutely loves Comedy movies and finds them excited and fast-paced."\n
    """
    few_shot_prompt = ("""Here are some examples for you:\n 
    Example one: The youth respondent gender is female,20years old, the, Young people's preferences for music and some movie types are as followsThe respondent absolutely loves music and finds it fascinating and enjoyable. The respondent feels neutral towards slow or fast songs, neither particularly liking nor disliking them. The respondent dislikes dance music and rarely listens to it. The respondent absolutely dislikes folk music and has no interest in it. The respondent dislikes country music and rarely listens to it. The respondent dislikes classical music and rarely listens to it. The respondent absolutely dislikes musical music and has no interest in it. The respondent absolutely loves pop music and finds it fun and energetic. The respondent absolutely loves rock music and finds it thrilling and powerful. The respondent absolutely dislikes metal or hard rock music and has no interest in it. The respondent absolutely dislikes punk music and has no interest in it. The respondent absolutely dislikes hip hop or rap music and has no interest in it. The respondent absolutely dislikes reggae or ska music and has no interest in it. The respondent absolutely dislikes swing or jazz music and has no interest in it. The respondent feels neutral towards rock n roll music, neither particularly liking nor disliking it. The respondent absolutely dislikes alternative music and has no interest in it. The respondent absolutely dislikes Latino music and has no interest in it. The respondent absolutely dislikes techno or trance music and has no interest in it. The respondent absolutely dislikes opera music and has no interest in it. The young respondent absolutely loves movies and finds them fascinating and entertaining. The young respondent likes horror movies and enjoys watching them occasionally. The young respondent likes romantic movies and enjoys watching them occasionally. The young respondent likes sci-fi movies and enjoys watching them occasionally. The young respondent absolutely dislikes war movies and has no interest in them. The young respondent absolutely loves fantasy or fairy tale movies and finds them magical and captivating. The young respondent absolutely loves animated movies and finds them imaginative and fun. These questions answers are: B,C,B,E\n
    Example two: The youth respondent gender is female,19years old, the, Young people's preferences for music and some movie types are as followsThe respondent absolutely loves music and finds it fascinating and enjoyable. The respondent feels neutral towards slow or fast songs, neither particularly liking nor disliking them. The respondent dislikes dance music and rarely listens to it. The respondent absolutely loves folk music and finds it relaxing and enjoyable. The respondent dislikes country music and rarely listens to it. The respondent dislikes classical music and rarely listens to it. The respondent absolutely loves musical music and finds it inspiring and uplifting. The respondent feels neutral towards pop music, neither particularly liking nor disliking it. The respondent absolutely loves rock music and finds it thrilling and powerful. The respondent dislikes metal or hard rock music and rarely listens to it. The respondent feels neutral towards punk music, neither particularly liking nor disliking it. The respondent dislikes hip hop or rap music and rarely listens to it. The respondent likes reggae or ska music and enjoys it from time to time. The respondent likes swing or jazz music and enjoys it occasionally. The respondent likes rock n roll music and enjoys it from time to time. The respondent likes alternative music and enjoys it occasionally. The respondent absolutely loves Latino music and finds it passionate and lively. The respondent absolutely dislikes techno or trance music and has no interest in it. The respondent dislikes opera music and rarely listens to it. The young respondent absolutely loves movies and finds them fascinating and entertaining. The young respondent dislikes horror movies and rarely watches them. The young respondent absolutely loves romantic movies and finds them heartwarming and touching. The young respondent absolutely dislikes sci-fi movies and has no interest in them. The young respondent feels neutral towards war movies, neither particularly liking nor disliking them. The young respondent likes fantasy or fairy tale movies and enjoys watching them occasionally. The young respondent likes animated movies and enjoys watching them occasionally. These questions answers are: B,D,A,E\n
    Example three: The youth respondent gender is female,17years old, the, Young people's preferences for music and some movie types are as followsThe respondent absolutely loves music and finds it fascinating and enjoyable. The respondent feels neutral towards slow or fast songs, neither particularly liking nor disliking them. The respondent absolutely dislikes dance music and has no interest in it. The respondent absolutely dislikes folk music and has no interest in it. The respondent absolutely dislikes country music and has no interest in it. The respondent likes classical music and enjoys it on occasion. The respondent absolutely dislikes musical music and has no interest in it. The respondent dislikes pop music and rarely listens to it. The respondent absolutely loves rock music and finds it thrilling and powerful. The respondent absolutely dislikes metal or hard rock music and has no interest in it. The respondent absolutely dislikes punk music and has no interest in it. The respondent absolutely dislikes hip hop or rap music and has no interest in it. The respondent absolutely dislikes reggae or ska music and has no interest in it. The respondent dislikes swing or jazz music and rarely listens to it. The respondent dislikes rock n roll music and rarely listens to it. The respondent absolutely loves alternative music and finds it unique and liberating. The respondent dislikes Latino music and rarely listens to it. The respondent absolutely dislikes techno or trance music and has no interest in it. The respondent dislikes opera music and rarely listens to it. The young respondent absolutely loves movies and finds them fascinating and entertaining. The young respondent feels neutral towards horror movies, neither particularly liking nor disliking them. The young respondent feels neutral towards romantic movies, neither particularly liking nor disliking them. The young respondent dislikes sci-fi movies and rarely watches them. The young respondent absolutely loves war movies and finds them intense and impactful. The young respondent absolutely loves fantasy or fairy tale movies and finds them magical and captivating. The young respondent absolutely loves animated movies and finds them imaginative and fun.These questions answers are: D,E,D,D\n
    Example four: The youth respondent gender is female,19years old, the, Young people's preferences for music and some movie types are as followsThe respondent absolutely loves music and finds it fascinating and enjoyable. The respondent dislikes slow or fast songs and rarely listens to them. The respondent feels neutral towards dance music, neither particularly liking nor disliking it. The respondent absolutely dislikes folk music and has no interest in it. The respondent absolutely dislikes country music and has no interest in it. The respondent likes classical music and enjoys it on occasion. The respondent feels neutral towards musical music, neither particularly liking nor disliking it. The respondent feels neutral towards pop music, neither particularly liking nor disliking it. The respondent absolutely loves rock music and finds it thrilling and powerful. The respondent absolutely loves metal or hard rock music and finds it energizing and intense. The respondent absolutely loves punk music and finds it rebellious and exciting. The respondent absolutely dislikes hip hop or rap music and has no interest in it. The respondent absolutely dislikes reggae or ska music and has no interest in it. The respondent absolutely dislikes swing or jazz music and has no interest in it. The respondent dislikes rock n roll music and rarely listens to it. The respondent feels neutral towards alternative music, neither particularly liking nor disliking it. The respondent dislikes Latino music and rarely listens to it. The respondent feels neutral towards techno or trance music, neither particularly liking nor disliking it. The respondent likes opera music and enjoys it from time to time. The young respondent absolutely loves movies and finds them fascinating and entertaining. The young respondent likes horror movies and enjoys watching them occasionally. The young respondent absolutely loves romantic movies and finds them heartwarming and touching. The young respondent dislikes sci-fi movies and rarely watches them. The young respondent dislikes war movies and rarely watches them. The young respondent dislikes fantasy or fairy tale movies and rarely watches them. The young respondent dislikes animated movies and rarely watches them.These questions answers are: D,C,B,E\n
    Example five: The youth respondent gender is male,20years old, the, Young people's preferences for music and some movie types are as followsThe respondent absolutely loves music and finds it fascinating and enjoyable. The respondent feels neutral towards slow or fast songs, neither particularly liking nor disliking them. The respondent absolutely loves dance music and finds it exciting and energizing. The respondent absolutely dislikes folk music and has no interest in it. The respondent absolutely dislikes country music and has no interest in it. The respondent absolutely dislikes classical music and has no interest in it. The respondent absolutely dislikes musical music and has no interest in it. The respondent feels neutral towards pop music, neither particularly liking nor disliking it. The respondent likes rock music and enjoys it from time to time. The respondent absolutely dislikes metal or hard rock music and has no interest in it. The respondent feels neutral towards punk music, neither particularly liking nor disliking it. The respondent absolutely loves hip hop or rap music and finds it rhythmic and expressive. The respondent dislikes reggae or ska music and rarely listens to it. The respondent absolutely dislikes swing or jazz music and has no interest in it. The respondent likes rock n roll music and enjoys it from time to time. The respondent feels neutral towards alternative music, neither particularly liking nor disliking it. The respondent dislikes Latino music and rarely listens to it. The respondent feels neutral towards techno or trance music, neither particularly liking nor disliking it. The respondent absolutely dislikes opera music and has no interest in it. The young respondent absolutely loves movies and finds them fascinating and entertaining. The young respondent feels neutral towards horror movies, neither particularly liking nor disliking them. The young respondent feels neutral towards romantic movies, neither particularly liking nor disliking them. The young respondent absolutely dislikes sci-fi movies and has no interest in them. The young respondent absolutely loves war movies and finds them intense and impactful. The young respondent likes fantasy or fairy tale movies and enjoys watching them occasionally. The young respondent likes animated movies and enjoys watching them occasionally These questions answers are: D,C,C,E\n
""")

   # question_prompt += few_shot_prompt
    prompt_format = "Your response should consist of just four letter separated by Three commas without any additional text or explanation. "
    return cond_info_prompt + question_prompt + prompt_format  # + "\n"



