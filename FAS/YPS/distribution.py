import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 加载数据集
def load_data(file_path):
    """加载数据集"""
    try:
        data = pd.read_csv(file_path)
        print("数据加载成功！")
        return data
    except Exception as e:
        print(f"加载数据时出错：{e}")
        return None
    
def preprocess_data(data):
    data['HHSEX'] = data['HHSEX'].map({1: 'Female', 2: 'Male'})
    bins = [17, 30, 40, 50, 60, 70, 80, 90]
    labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']  
    data['HHAGE_group'] = pd.cut(data['HHAGE'], bins=bins, labels=labels, right=False)
    employ = {1: 'Employed full-time',
              2: 'Employed part-time',
              3:'Retired',
              4:'Not employed'}
    data['EMPLOYHH'] = data['EMPLOYHH'].map(employ)
    state_mapping = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
        'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
        'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
        'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
        'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
        'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
        'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    data['state_postal'] = data['state_postal'].map(state_mapping)
    return data

def plot_violin_with_values(data, column, save_path):
    counts = data[column].value_counts().sort_index()
    total = counts.sum()
    print(f"\n{column} 的具体数值和比例：")
    for idx, value in counts.items():
        print(f"{idx}: {value} ({value / total:.2%})")
    
    # 绘制风琴图
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x=data[column], density_norm='width', inner=None, color="lightblue")
    
    for i, value in enumerate(counts.values):
        plt.text(i, value, f"{value}\n({value / total:.2%})", ha='center', va='bottom', fontsize=10, color='black')
    
    # 移除标题、x轴和y轴标签
    plt.title('')  # 移除标题
    plt.xlabel('')  # 移除x轴标签
    plt.ylabel('')  # 移除y轴标签
    
    # 移除x轴和y轴的刻度标签
    plt.xticks([])  # 移除x轴刻度标签
    plt.yticks([])  # 移除y轴刻度标签
    
    # 调整边距，防止标签被截断
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # 确保结果目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存图表
    plt.savefig(os.path.join(save_path, f"{column}_distribution_violin.png"), transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# 统计指定列的特征分布
def summarize_features(data, columns, save_path):
    """统计指定列的特征分布"""
    for column in columns:
        print(f"\n分析 {column} 的分布情况：")
        plot_violin_with_values(data, column, save_path)

# 主函数
if __name__ == "__main__":
    file_path = "D:/HaolingX/Role-Playing LLM+Survey/code/Full_Simulation/data_prepro/Data/selected_RECS.csv"  # 替换为你的数据集路径
    result_path = "Result"
    data = load_data(file_path)
    if data is not None:
        columns_to_analyze = ["HHSEX", "HHAGE_group", "EMPLOYHH", "state_postal"]
        data = preprocess_data(data)
        summarize_features(data, columns_to_analyze, result_path)
