import pandas as pd
import numpy as np
from numpy import percentile

def map_gender(gender_value):
    if gender_value == 1:
        return 'male'
    elif gender_value == 2:
        return 'female'
    else:
        return 'unknown'
def map_age(age_group):
    if age_group == 1:
        return 'less than 18 years old.'
    elif age_group == 2:
        return 'between the ages of 18 and 24.'
    elif age_group == 3:
        return 'between the ages of 24 to 30.'
    elif age_group == 4:
        return 'more than 30 years old.'
    else:
        return 'unknown'
def map_tier(tier):
    if tier == 1:
        return "first-tier city"
    elif tier == 2:
        return "sencond-tier city"
    elif tier ==3 :
        return "third-tier city"
    else:
        return 'unknown'
def main():
    file_path = './Data/Trell social media usage/'
    input_path = file_path + 'filtered.csv'
    output_path = file_path + 'selected_features.csv'

    df = pd.read_csv(input_path)
    mapping_features = {
        'gender': map_gender,
        'age_group': map_age,
        'tier':map_tier,
    }
    transform_col = ['gender','age_group','tier'] 
    for col in transform_col:
        if col in mapping_features:
            df[col] = df[col].apply(mapping_features[col])
    print("ok1")
    # extract some features
    extract_features = ['userId','tier','gender','age_group','following_rate',
                       'number_of_words_per_action','avgCompletion','avgTimeSpent','avgDuration',
                       'creations','content_views',
                       'weekends_trails_watched_per_day','weekdays_trails_watched_per_day',
                       'slot1_trails_watched_per_day','slot2_trails_watched_per_day','slot3_trails_watched_per_day','slot4_trails_watched_per_day',
                       'avgt2']
    
    subset_df = df[extract_features].copy()
    subset_df.to_csv(output_path, index=False)
    print("ok2")
    # sd = df['gender']
    # selected_data = df[
    #     'userId','tier','gender',
    #     'number_of_words_per_action','avgCompletion','avgTimeSpent','avgDuration']
    #print(selected_data)

if __name__ == "__main__":
    main()