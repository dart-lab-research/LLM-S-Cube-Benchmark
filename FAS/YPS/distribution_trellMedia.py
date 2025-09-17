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
    data['gender'] = data['gender'].map({'male': 'Male', 'female': 'Female'})
    # 处理 tier
    tier_mapping = {
        'first-tier city': 'First-tier City',
        'second-tier city': 'Second-tier City',
        'third-tier city': 'Third-tier City'
    }
    data['tier'] = data['tier'].map(tier_mapping)

    age_group_mapping = {
        'less than 18 years old.': 'Less than 18',
        'between the ages of 18 and 24.': '18-24',
        'between the ages of 24 to 30.': '24-30', 
        'more than 30 years old.': 'More than 30'
    }
    data['age_group'] = data['age_group'].map(age_group_mapping)

    
    return data

def plot_bar_with_values(data, column, title, save_path):
    counts = data[column].value_counts().sort_index()
    total = counts.sum()
    print(f"\n{title} 的具体数值和比例：")
    for idx, value in counts.items():
        print(f"{idx}: {value} ({value / total:.2%})")
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    for i, value in enumerate(counts.values):
        plt.text(i, value, f"{value}\n({value / total:.2%})", ha='center', va='bottom', fontsize=10, color='black')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{column}_distribution.png"))
    plt.show()

# 统计指定列的特征分布
def summarize_features(data, columns, save_path):
    """统计指定列的特征分布"""
    # print("数据集基本信息：")
    # print(data.info())
    # print("\n指定列的描述性统计：")
    # print(data[columns].describe(include='all'))

    # 遍历指定列，绘制分布图
    for column in columns:
        print(f"\n分析 {column} 的分布情况：")
        plot_bar_with_values(data, column, f"{column} Distribution", save_path)

# 主函数
if __name__ == "__main__":
    file_path = "data_prepro\\Data\\random_features.csv"  # 替换为你的数据集路径
    result_path = "data_prepro\\Result\\TrellMedia"
    data = load_data(file_path)
    if data is not None:
        # 指定需要分析的列
        columns_to_analyze = ["gender", "tier", "age_group"]
        data = preprocess_data(data)
        summarize_features(data, columns_to_analyze, result_path)