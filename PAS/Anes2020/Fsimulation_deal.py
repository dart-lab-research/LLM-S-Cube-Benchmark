import csv  
import pandas as pd

folder_path = './Data/Anes2020/'
# Read the data from the text file  
input_file = folder_path + 'a20_descriptions_fsimulation_Given.txt' 
output_file = folder_path + "anes_fsimulation_Given.csv"  


def get_generatecsv():
    # Initialize a list to store rows  
    rows = []  

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
    column_names = ["trump", "obama", "biden"]  # Replace with your desired column names  

    # Write the rows into a CSV file  
    with open(output_file, "w", newline="") as csvfile:  
        writer = csv.writer(csvfile)  
        # Write the column names first  
        writer.writerow(column_names)  
        # Write the data rows  
        writer.writerows(rows)  

    print(f"Data has been successfully written to {output_file}")  

def merge():
    # 定义输入和输出文件路径  
    input_file1 = "./Data/Anes2020/anes_fsimulation_Given.csv"   # 第一个CSV文件  
    input_file2 = folder_path + 'selected_anes2020.csv'  # 第二个CSV文件  
    output_file = folder_path+"merged_output_Given.csv"  # 合并后的CSV文件  

    columns_from_file1 = ["trump", "obama", "biden"]  # 从第一个文件中抽取的列  
    columns_from_file2 = ["fttrump1", "ftobama1","ftbiden1"]  # 从第二个文件中抽取的列  

    # 定义新的列名  
    new_columns_file1 = {"trump": "ftrump", "obama": "fobama","biden":"fbiden"}  # 第一个文件的列重命名  
    new_columns_file2 = {"fttrump1":"rtrump", "ftobama1":"robama","ftbiden1":"rbiden"}  # 第二个文件的列重命名  

    # 读取第一个CSV文件并抽取指定列  
    df1 = pd.read_csv(input_file1, usecols=columns_from_file1)  
    # 重命名列  
    df1.rename(columns=new_columns_file1, inplace=True)  

    # 读取第二个CSV文件并抽取指定列  
    df2 = pd.read_csv(input_file2, usecols=columns_from_file2)  
    # 重命名列  
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

if __name__ == "__main__":
    get_generatecsv()
    merge()