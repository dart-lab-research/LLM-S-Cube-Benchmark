import pandas as pd
import csv

#Average number of words used by the user per comment.
def map_commentNum(number_of_words_per_action):
    number_of_words_per_action = float(number_of_words_per_action)
    if 0 <= number_of_words_per_action < 0.2:
        return f'comments with very few words(normalized score {number_of_words_per_action}, in range of [0,0.2])'
    elif 0.2 <= number_of_words_per_action < 0.4:
        return f'comments with fewer words(normalized score {number_of_words_per_action}, in range of [0.2,0.4])'
    elif 0.4 <= number_of_words_per_action < 0.6:
        return f'comment using moderate sized words(normalized score {number_of_words_per_action}, in range of [0.4,0.6])'
    elif 0.6 <= number_of_words_per_action < 0.8:
        return f'uses more words in comments(normalized score {number_of_words_per_action}, in range of [0.6,0.8])'
    elif 0.8 <= number_of_words_per_action <= 1.0:
        return f'comments with many words(normalized score {number_of_words_per_action}, in range of [0.8,1])'
    else:
        return 'unknown'
    
#Total number of videos uploaded by the user.
def map_creation(creation):
    creation = float(creation)
    if 0 <= creation < 0.2:
        return f'uploads very few videos(normalized score {creation}, in range of [0,0.2])'
    elif 0.2 <= creation < 0.4:
        return f'uploads fewer videos(normalized score {creation}, in range of [0.2,0.4])'
    elif 0.4 <= creation < 0.6:
        return f'uploads some videos(normalized score {creation}, in range of [0.4,0.6])'
    elif 0.6 <= creation < 0.8:
        return f'uploads more videos(normalized score {creation}, in range of [0.6,0.8])'
    elif 0.8 <= creation <= 1.0:
        return f'uploads lots of videos(normalized score {creation}, in range of [0.8,1]'
    else:
        return 'unknown'
    
#Total number of videos watched.>1
def map_contentViews(content_views):
    content_views = float(content_views)
    if 0 <= content_views < 0.2:
        return f'watches very few videos(normalized score {content_views}, in range of [0,0.2])'
    elif 0.2 <= content_views < 0.4:
        return f'watches fewer videos(normalized score {content_views}, in range of [0.2,0.4])'
    elif 0.4 <= content_views < 0.6:
        return f'watches some videos(normalized score {content_views}, in range of [0.4,0.6])'
    elif 0.6 <= content_views < 0.8:
        return f'watches more videos(normalized score {content_views}, in range of [0.6,0.8])'
    elif 0.8 <= content_views <= 1.0:
        return f'watches lots of videos(normalized score {content_views}, in range of [0.8,1])'
    else:
        return f'watches plenty of videos(normalized score {content_views}, in range of [1,..])'

#weekends_trails_watched_per_day
def map_weekends_trails(wtwpd):
    wtwpd = float(wtwpd)
    if 0 <= wtwpd < 0.2:
        return f'and very few videos on weekends per day(normalized score {wtwpd}, in range of [0,0.2])'
    elif 0.2 <= wtwpd < 0.4:
        return f'and fewer videos on weekends per day(normalized score {wtwpd}, in range of [0.2,0.4])'
    elif 0.4 <= wtwpd < 0.6:
        return f'and some videos on weekends per day(normalized score {wtwpd}, in range of [0.4,0.6])'
    elif 0.6 <= wtwpd < 0.8:
        return f'and more videos on weekends per day(normalized score {wtwpd}, in range of [0.6,0.8])'
    elif 0.8 <= wtwpd <= 1.0:
        return f'and lots of videos on weekends per day(normalized score {wtwpd}, in range of [0.8,1])'
    else:
        return 'unknown'

#weekdays_trails_watched_per_day
def map_weekdays_trails(wtwpd):
    wtwpd = float(wtwpd)
    if 0 <= wtwpd < 0.2:
        return f'watches very few videos on weekdays per day(normalized score {wtwpd}, in range of [0,0.2])'
    elif 0.2 <= wtwpd < 0.4:
        return f'watches fewer videos on weekdays per day(normalized score {wtwpd}, in range of [0.2,0.4])'
    elif 0.4 <= wtwpd < 0.6:
        return f'watches some videos on weekdays per day(normalized score {wtwpd}, in range of [0.4,0.6])'
    elif 0.6 <= wtwpd < 0.8:
        return f'watches more videos on weekdays per day(normalized score {wtwpd}, in range of [0.6,0.8])'
    elif 0.8 <= wtwpd <= 1.0:
        return f'watches lots of videos on weekdays per day(normalized score {wtwpd}, in range of [0.8,1])'
    else:
        return 'unknown'
    
#The day is divided into 4 slots. This feature represents the average number of videos watched in this particular time slot.
def map_slot(slot, i):
    slot = float(slot)
    if 0 <= slot < 0.2:
        return f'the average number of videos watched in time slot{i} is very small(normalized score {slot}, , in range of [0,0.2])'
    elif 0.2 <= slot < 0.4:
        return f'the average number of videos watched in time slot{i} is small(normalized score {slot}, , in range of [0.2,0.4])'
    elif 0.4 <= slot < 0.6:
        return f'the average number of videos watched in time slot{i} is moderate(normalized score {slot}, , in range of [0.4,0.6])'
    elif 0.6 <= slot < 0.8:
        return f'the average number of videos watched in time slot{i} is considerable(normalized score {slot}, , in range of [0.6,0.8])'
    elif 0.8 <= slot <= 1.0:
        return f'the average number of videos watched in time slot{i} is very large(normalized score {slot}, , in range of [0.8,1])'
    else:
        return 'unknown'

def generate_description(row):
    #basic info
    tier = row['tier']
    gender = row['gender']
    age_group = row['age_group']

    #usage rate,time spent(second)
    completion = row['avgCompletion']
    timeSpent = row['avgTimeSpent']
    duration = row['avgDuration']

    #normalized info
    # 
    # h_size = row['HHSIZE']
    # hh_income = row['HHFAMINC']
    # home_own = 'own their home' if row['HOMEOWN'] == 1 else 'rent'
    # h_veh_cnt = row['HHVEHCNT']
    # age = row['age']
    # gender = 'male' if row['gender'] == 1 else 'female'
    
    # description1 = f"In a {h_size}-person family with an income of {hh_income}, who {home_own} and has {h_veh_cnt} cars, "
    description_time= f'''
    This one's average watch time completion rate of the videos is {completion},
      and the average time spent on a video in seconds is {timeSpent}.
      Average duration of the videos that this person has watched till date is {duration}."
    '''
    description_Nminfo = f""
    description_basInfo = f"This one has an age group {age_group}, who lives in a {tier}."
    
    return description_basInfo+description_Nminfo+description_time

def generate_usageinfo(row):
    numPerAction = map_commentNum(row['number_of_words_per_action'])
    creation = map_creation(row['creations'])
    conView = map_contentViews(row['content_views'])
    WeekendsWatch = map_weekends_trails(row['weekends_trails_watched_per_day'])
    WeekdaysWatch = map_weekdays_trails(row['weekdays_trails_watched_per_day'])
    slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
    slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
    slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
    slot4 = map_slot(row['slot4_trails_watched_per_day'],4)

    description_Usageinfo = f'''
    When talk about Trell usage, this person {conView}.
    Besides, this person {WeekendsWatch},while at the same time {numPerAction} and {creation}.
    '''
    return description_Usageinfo
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Trell social media usage/'
    input_path = folder_path + 'selected_features.csv'
    output_path = folder_path + 'media_descriptions.txt'
    df = pd.read_csv(input_path) 
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        for row in reader:
            description = generate_description(row)
            usageinfo = generate_usageinfo(row)
            print(description + '\n' + usageinfo)
            break
            #outfile.write(description + '\n\n')
            cnt = cnt + 1
            if cnt>1000:
                break
            
    
    # descriptions = df.apply(generate_description, axis=1)

    # # 将生成的描述性字符串保存到新的DataFrame中
    # result_df = pd.DataFrame({'Description': descriptions})

    # with open('output_descriptions.txt', 'w') as file:
    #     for description in descriptions:
    #         file.write(description + '\n')
if __name__ == "__main__":
    main()