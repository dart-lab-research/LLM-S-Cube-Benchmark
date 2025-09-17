import pandas as pd
import numpy as np
from numpy import percentile


def age_group_detect(age_group):
    Clean_Again = ['avgTimeSpent', 'content_views' , 'creations', 'avgt2', 'number_of_words_per_action' , 'avgTimeSpent']
    
    for x in Clean_Again:
        #print('column name:', x)
        upper_limit = age_group[x].mean() + 2 * age_group[x].std()
        lower_limit = age_group[x].mean() - 2 * age_group[x].std()

        outliers = age_group[(age_group[x]>upper_limit) | (age_group[x]<lower_limit)] #delete by upper/lower limit
        #print(len(outliers))

        age_group.drop(outliers.index , axis = 0 , inplace= True)
        #print(age_group.shape[0])

    return age_group

def data_detect(file_path):
    df = pd.read_csv(file_path + 'train_age_dataset.csv')
    df = df.drop('Unnamed: 0' , axis = 1)

    #print(df.columns)
    #print(df['tier'].describe())
    q25, q75 = percentile(df.loc[:,'age_group'], 25), percentile(df.loc[:,'num_of_hashtags_per_action'], 75)
    iqr = q75 - q25

    #delte hashtags>0.4 length = 66
    outliers = [x for x in df.loc[:,'num_of_hashtags_per_action'] if x > 0.4]
    outliers = df.query('@outliers in num_of_hashtags_per_action')
    df_filtered = df.drop(outliers.index , axis = 0)
    
    #delete agegroup1 to 4
    df_filtered_age_group1 = pd.DataFrame(df_filtered[df_filtered['age_group'] == 1])
    df_filtered_age_group1 = age_group_detect(df_filtered_age_group1)
    df_filtered_age_group2 = pd.DataFrame(df_filtered[df_filtered['age_group'] == 2])
    df_filtered_age_group2 = age_group_detect(df_filtered_age_group2)
    df_filtered_age_group3 = pd.DataFrame(df_filtered[df_filtered['age_group'] == 3])
    df_filtered_age_group3 = age_group_detect(df_filtered_age_group3)
    df_filtered_age_group4 = pd.DataFrame(df_filtered[df_filtered['age_group'] == 4])
    df_filtered_age_group4 = age_group_detect(df_filtered_age_group4)

    df = pd.concat([df_filtered_age_group1 , df_filtered_age_group2 , df_filtered_age_group3 , df_filtered_age_group4], axis=0)
    print(df.shape)
    df.to_csv(file_path +'filtered.csv', index=False)  
def main():
    file_path = './Data/Trell social media usage/'
    input_path = file_path + 'train_age_dataset.csv'
    output_path = file_path + 'selected_features.csv'
    data_detect(file_path)
    #df.to_csv('new_file_path.csv', index=False)  

if __name__ == "__main__":
    main()