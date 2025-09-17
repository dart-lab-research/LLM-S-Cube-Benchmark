import pandas as pd
import numpy as np
from numpy import percentile

def main():
    file_path = 'Data/RECS/'
    input_path = '/home/cyyuan/Data/RECS/recs2020_public_v7.csv'
    output_path = "/home/cyyuan/Data/RECS/selected_RECS.csv"

    df = pd.read_csv(input_path)

    #
    life_course_features = ['HHSEX', 'HHAGE', 'EMPLOYHH', 'state_postal','EDUCATION', 'HOUSEHOLDER_RACE', 'NHSLDMEM', 'ATHOME', 'MONEYPY']
    
    #
    fuel_features = ['HDD65', 'CDD65', 'UGASHERE',"ELFOOD",'LPCOOK','UGCOOK']

    #usage
    use_features = ['USEEL', 'USENG','USELP','USEFO','USESOLAR','USEWOOD',"ALLELEC"]

    #COST
    cost_features = ['KWH', "DOLLAREL","TOTALBTU","TOTALDOL"]
    
    

    # extract some features
    extract_features = life_course_features + fuel_features + use_features + cost_features
    
    subset_df = df[extract_features].copy()
    subset_df.to_csv(output_path, index=False)
    print("ok2")


if __name__ == "__main__":
    main()