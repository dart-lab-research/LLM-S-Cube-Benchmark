import numpy as np
import csv
with open(r"/home/cyyuan/Data/GSS/gpt/4accfew.csv", mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        for i,row in enumerate(reader):
                if row["responses"] == row["gts"]:
                        cnt= cnt +1

        print(cnt/(i))
# import pandas as pd  
# import numpy as np  
# from scipy.stats import entropy  

# # 读取CSV文件  
# def calculate_kl_divergence(csv_file, col1, col2, num_rows=500):  
#     # 读取CSV文件并取前 num_rows 条数据  
#     data = pd.read_csv(csv_file).head(num_rows)  
    
#     # 提取两列数据  
#     p = data[col1].values  
#     q = data[col2].values  
    
#     # 添加平滑值以避免零值问题  
#     epsilon = 1e-10  
#     p = p + epsilon  
#     q = q + epsilon  
    
#     # 归一化为概率分布  
#     p = p / np.sum(p)  
#     q = q / np.sum(q)  
    
#     # 计算KL散度  
#     kl_div = entropy(p, q)  # scipy.stats.entropy 计算 KL 散度  
#     return kl_div 

# # 示例用法  
# csv_file = "/home/cyyuan/Data/RECS/responses/numerical_dollar.csv"  # 替换为你的CSV文件路径  
# col1 = "responses"  # 替换为你的第一列列名  
# col2 = "gts"  # 替换为你的第二列列名  

# kl_divergence = calculate_kl_divergence(csv_file, col1, col2, num_rows = 500)  
# print(f"KL散度: {kl_divergence}")