import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

# 设置随机种子以便重现  
np.random.seed(42)  

# 生成数据  
group_a = np.random.normal(loc=5, scale=1, size=100)  
group_b = np.random.normal(loc=7, scale=1.5, size=100)  

# 创建DataFrame  
data = pd.DataFrame({  
    'Group': ['A'] * 100 + ['B'] * 100,  
    'Value': np.concatenate([group_a, group_b])  
})  

# 绘制小提琴图  
plt.figure(figsize=(10, 6))  
sns.violinplot(x='Group', y='Value', data=data)  
plt.title('Violin Plot of Two Groups')  
plt.xlabel('Group')  
plt.ylabel('Value')  
plt.grid(True)  
plt.savefig("a.jpg")
plt.show()