import csv  
import random  

folder_path = './Data/Trell social media usage/'
input_path = folder_path + 'selected_features.csv'
output_path = folder_path + 'random_features.csv'

# 读取 CSV 文件的所有行  
with open(input_path, mode='r', newline='') as infile:  
    reader = csv.DictReader(infile)  
    rows = list(reader)  # 将所有行转换为列表  

# 随机打乱行的顺序  
random.shuffle(rows)  

# 将随机打乱的行写入新的 CSV 文件  
with open(output_path, mode='w', newline='') as outfile:  
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)  
    writer.writeheader()  # 写入表头  
    writer.writerows(rows)  # 写入打乱后的行

print('ok')