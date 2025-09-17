import pandas as pd
import numpy as np
from numpy import percentile

def main():
    
    file_path = './Data/Anes2020/'
    input_path = file_path + 'anes_pilot_2020ets_csv.csv'
    output_path = file_path + 'selected_anes2020.csv'

    df = pd.read_csv(input_path)

    #年龄，hh, family income, vote20turnoutjb是否参与投票, 参与度汇总分数
    life_course_features = ['age','home_ownership','income','vote20turnoutjb','particip_count']
    
    #政治参与
    enrollment_features = ['meeting','moneyorg','protest','online', 'persuade','button']

    #社会和文化态度
    sp_features = ['loans2', 'diversity7', 'experts', 'compro1', 'compro2', 'pcorrect']
    
    #行为表现
    behave_features = ['expavoid', 'callout_social', 'callout_person']
    
    #对政客评分 mike pence/andrew yang/pelosi/
    politician_features = ['fttrump1', 'ftbiden1', 'ftobama1','ftpence1', 'ftyang1', 'ftpelosi1', 'ftrubio1','ftocasioc1', 'fthaley1','ftthomas1', 'ftfauci1']
    

    # extract some features
    extract_features = life_course_features+enrollment_features+sp_features+behave_features+politician_features
    
    subset_df = df[extract_features].copy()
    subset_df.to_csv(output_path, index=False)
    print("ok2")



    # 读取CSV文件  
    df = pd.read_csv(output_path)   
    df = df[~((df['home_ownership'] == 9) | (df['vote20turnoutjb'] == 9) | (df['ftpence1'] == 999)| (df['ftyang1'] == 999)| (df['ftpelosi1'] == 999)| (df['ftrubio1'] == 999)| (df['ftocasioc1'] == 999)| (df['fthaley1'] == 999)| (df['ftthomas1'] == 999)| (df['ftfauci1'] == 999)| (df['fttrump1'] == 999)| (df['ftbiden1'] == 999)| (df['ftobama1'] == 999))]  

    # 保存修改后的数据到新的CSV文件  
    df.to_csv(output_path, index=False)
    print("okk")

    shuffled_df = df.sample(frac=1).reset_index(drop=True)  

    shuffled_df.to_csv(output_path, index=False) 
    print('shuffled')
if __name__ == "__main__":
    main()