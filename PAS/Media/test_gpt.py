import pandas as pd  
import numpy as np  

# 读取 CSV 文件  
file_path = r"/home/cyyuan/Data/Trell social media usage/merged_features_3.csv"  # 替换为您的 CSV 文件路径  
data = pd.read_csv(file_path)  

# 提取需要计算的列  
fconviews = data["fconviews"].values  
rconviews = data["rconviews"].values  

# 定义 KL 散度计算函数  
def kl_divergence(p, q):  
    # 确保分布归一化  
    p = p / np.sum(p)  
    q = q / np.sum(q)  
    # 添加平滑项，防止零值问题  
    epsilon = 1e-10  
    p = np.clip(p, epsilon, None)  
    q = np.clip(q, epsilon, None)  
    # 计算 KL 散度  
    return np.sum(p * np.log(p / q))  

# 计算 fconviews 和 rconviews 的 KL 散度  
kl_fconviews_rconviews = kl_divergence(fconviews, rconviews)  
print("KL 散度 (fconviews || rconviews):", kl_fconviews_rconviews)  

# 如果需要计算其他列的 KL 散度，例如 fweekends 和 rweekends  
fweekends = data["fweekends"].values  
rweekends = data["rweekends"].values  
kl_fweekends_rweekends = kl_divergence(fweekends, rweekends)  
print("KL 散度 (fweekends || rweekends):", kl_fweekends_rweekends)  

# 计算 fweekdays 和 rweekdays 的 KL 散度  
fweekdays = data["fweekdays"].values  
rweekdays = data["rweekdays"].values  
kl_fweekdays_rweekdays = kl_divergence(fweekdays, rweekdays)  
print("KL 散度 (fweekdays || rweekdays):", kl_fweekdays_rweekdays)  