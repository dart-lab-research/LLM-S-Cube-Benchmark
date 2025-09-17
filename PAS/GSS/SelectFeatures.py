import pandas as pd
import numpy as np
from numpy import percentile

def main():
    file_path = './Data/GSS/'
    input_path = file_path + 'GSS2022.csv'
    output_path = file_path + 'selected_GSS.csv'

    df = pd.read_csv(input_path)

    #ID,年龄，性别 人种，兄弟姐妹，婚姻状况，孩子数量，地区，教育程度，宗教信仰,收入
    life_course_features = ['id','age','sex','racecen1','sibs','marital','childs','region','educ','relig','income']
    
    #工作状态，所在单位，家庭构成，household
    employment_features = ['wrkstat','occ10','hhtype1','hompop']

    #国家政策
    #自由主义->极端主义,环保，国家健康，解决大城市问题,遏制犯罪率上升,改善教育系统，军队支出，福利，社会安全
    political_features = ['polviews', 'natenvir','natheal','natcity','natcrime','nateduc','natarms','natfare','natsoc']
    

    # extract some features
    extract_features = life_course_features+employment_features+political_features
    
    subset_df = df[extract_features].copy()
    subset_df.to_csv(output_path, index=False)
    print("ok2")


if __name__ == "__main__":
    main()