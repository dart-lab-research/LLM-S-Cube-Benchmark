import csv  


# 原始数据文件路径  
input_file = 'D:\\School\\UM\\LLM\\Media\\3.txt'  
output_file = 'D:\\School\\UM\\LLM\\Media\\gen3.csv'  # 输出的 CSV 文件路径  

# 表头  
headers = [  
    "tier",  
    "gender",  
    "age_group",  
    "content_views",  
    "weekends_trails_watched_per_day",  
    "weekdays_trails_watched_per_day"  
]  

# 处理数据并写入 CSV 文件  
def process_data(input_file, output_file):  
    with open(input_file, 'r', encoding='utf-8') as infile:  
        lines = infile.readlines()  
    
    # 清理数据  
    cleaned_data = []  
    for line in lines:  
        line = line.strip()  # 去掉首尾空格和换行符  
        if line.endswith(","):  # 去掉行尾多余的逗号  
            line = line[:-1]  
        if line.startswith("[") and line.endswith("]"):  # 确保是列表格式  
            line = eval(line)  # 将字符串转换为 Python 列表  
            cleaned_data.append(line)  
    
    # 写入 CSV 文件  
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:  
        writer = csv.writer(csvfile)  
        writer.writerow(headers)  # 写入表头  
        writer.writerows(cleaned_data)  # 写入数据  

# 调用函数  
process_data(input_file, output_file)  
print(f"数据已成功写入到 {output_file}")
# # 读取 txt 文件并写入 csv 文件  
# def txt_to_csv(txt_file, csv_file):  
#     with open(txt_file, 'r', encoding='utf-8') as txt_f:  
#         lines = txt_f.readlines()  
    
#     # 处理每一行数据，去掉多余的空格和换行符  
#     data = [eval(line.strip()) for line in lines]  # 使用 eval 将字符串转换为列表  

#     # 写入到 CSV 文件  
#     with open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:  
#         writer = csv.writer(csv_f)  
#         # 写入表头  
#         writer.writerow(["tier", "gender", "age_group", "content_views", "weekends_trails_watched_per_day", "weekdays_trails_watched_per_day"])  
#         # 写入数据  
#         writer.writerows(data)  

# # 输入和输出文件路径  
# txt_file = 'D:\\School\\UM\\LLM\\Media\\100.txt'  # 替换为你的 txt 文件路径  
# csv_file = '100.csv'  # 替换为你想要保存的 csv 文件路径  

# # 调用函数  
# txt_to_csv(txt_file, csv_file)