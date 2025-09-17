import csv  
import pandas as pd
from plot import calculate_kl_divergence

folder_path = './Data/GSS/'
# Read the data from the text file  
input_file = folder_path + 'GSS_fsimulation.txt' 
output_file = folder_path + "GSS_fsimulation.csv"  


def get_generatecsv():
    # Initialize a list to store rows  
    rows = []  
    folder_path = './Data/GSS/'
# Read the data from the text file  
    input_file = r"/home/cyyuan/Data/RECS/gpt/gptzero4.txt"
    output_file = r'/home/cyyuan/Data/RECS/gptzero4.csv'  
    # Read the text file line by line  
    with open(input_file, "r") as file:  
        for line in file:  
            # Remove any extra spaces or newline characters  
            line = line.strip()  
            # Convert the line into a list of values  
            # Remove brackets and split by commas  
            if line.startswith("[") and line.endswith(","):  
                line = line[1:-2]  # Remove the outer brackets  
                values = [value.strip().strip("'").strip('"') for value in line.split(",")]  
                rows.append(values)  

    # Write the rows into a CSV file  
    column_names = ['KWH', 'DOLLAREL', 'TOTALDOL']  # Replace with your desired column names  

    # Write the rows into a CSV file  
    with open(output_file, "w", newline="") as csvfile:  
        writer = csv.writer(csvfile)  
        # Write the column names first  
        writer.writerow(column_names)  
        # Write the data rows  
        writer.writerows(rows)  

    print(f"Data has been successfully written to {output_file}")  

def drop():
    input_file1 = "./Data/GSS/selected_GSS.csv"
    col = ['natheal','natcity','natenvir','natcrime','nateduc','natarms']
    df = pd.read_csv(input_file1,usecols=col)
    df.dropna(subset=col,inplace=True)
    df.to_csv("./Data/GSS/1.csv")

def merge():
    # 定义输入和输出文件路径  
    input_file1 = r"/home/cyyuan/Data/RECS/gptzero4.csv"   # 第一个CSV文件  
    input_file2 = r"/home/cyyuan/Data/RECS/selected_RECS.csv"  # 第二个CSV文件  
    output_file = r"/home/cyyuan/Data/RECS/gpt/gptzero4merged.csv"  # 合并后的CSV文件  

    columns_from_file1 = ['KWH', 'DOLLAREL', 'TOTALDOL']  # 从第一个文件中抽取的列  
    columns_from_file2 = ['KWH', 'DOLLAREL', 'TOTALDOL']  # 从第二个文件中抽取的列  

    # 定义新的列名  
    new_columns_file1 = {'KWH':"FKWH", 'DOLLAREL':'FDOLLAREL', 'TOTALDOL':"FTOTALDOL"}  # 第一个文件的列重命名  
    new_columns_file2 = {'KWH':"RKWH", 'DOLLAREL':'RDOLLAREL', 'TOTALDOL':"RTOTALDOL"}
    # 读取第一个CSV文件并抽取指定列  
    df1 = pd.read_csv(input_file1, usecols=columns_from_file1)  
    # 重命名列  
    df1.rename(columns=new_columns_file1, inplace=True)  

    # 读取第二个CSV文件并抽取指定列  
    df2 = pd.read_csv(input_file2, usecols=columns_from_file2)  
    df2.rename(columns=new_columns_file2, inplace=True) 

   
    # 获取两个文件的最短长度  
    min_length = min(len(df1), len(df2))  

    # 截取两个数据框到最短长度  
    df1 = df1.head(min_length)  
    df2 = df2.head(min_length)  

    # 合并两个数据框  
    merged_df = pd.concat([df1, df2], axis=1)  

    # 将合并后的数据写入新的CSV文件  
    merged_df.to_csv(output_file, index=False)  

    print(f"数据已成功合并并保存到 {output_file}")  

def cal():
    input_file = r"/home/cyyuan/Data/Trell social media usage/gpt/4fewmerged.csv"
    with open(input_file,'r') as infile:
        reader = csv.DictReader(infile)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0

        for i, row in enumerate(reader):
            if row['fenvir'] == row['renvir']:
                cnt1 = cnt1 + 1
            if row['fheal'] == row['rheal']:
                cnt2 = cnt2 + 1
            if row['fcity'] == row['rcity']:
                cnt3 = cnt3 + 1
        acc1 = cnt1/(i+1)
        acc2 = cnt2/(i+1)
        acc3 = cnt3/(i+1)

        print(acc1,acc2,acc3)
    return
import numpy as np

def kl_divergence(p, q):  
    # 避免 q 为 0 的情况  
    q = np.where(q == 0, 1e-10, q)  
    # 计算 KL 散度，同时避免无效值  
    return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))  
def calculate_kl_for_columns(file_path):  
    # 读取 CSV 文件  
    df = pd.read_csv(file_path)  

    # 获取所有 f 和 r 开头的列  
    f_columns = [col for col in df.columns if col.startswith('F')]  
    r_columns = [col for col in df.columns if col.startswith('R')]  

    # 确保 f 和 r 列数量一致  
    if len(f_columns) != len(r_columns):  
        raise ValueError("f 和 r 开头的列数量不一致！")  

    # 遍历每对 f 和 r 列，计算 KL 散度  
    kl_results = {}  
    for f_col, r_col in zip(f_columns, r_columns):  
        # 统计频率分布  
        f_counts = df[f_col].value_counts(normalize=True).sort_index()  
        r_counts = df[r_col].value_counts(normalize=True).sort_index()  

        # 确保两个分布的索引一致（补零处理）  
        all_indices = sorted(set(f_counts.index).union(set(r_counts.index)))  
        f_probs = np.array([f_counts.get(i, 0) for i in all_indices])  
        r_probs = np.array([r_counts.get(i, 0) for i in all_indices])  

        # 计算 KL 散度  
        kl_f_to_r = kl_divergence(f_probs, r_probs)  
        kl_r_to_f = kl_divergence(r_probs, f_probs)  

        # 保存结果  
        kl_results[f"{f_col} -> {r_col}"] = kl_f_to_r  


    # 打印结果  
    for key, value in kl_results.items():  
        print(f"{key}: {value}")  

# 文件路径  
if __name__ == "__main__":
     get_generatecsv()
     merge()
    #calculate_kl_for_columns(r"/home/cyyuan/RECS/ol3fewmerged.csv")
    # drop()