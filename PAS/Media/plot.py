import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from metrics import calculate_distribution_metrics, calculate_kl_divergence, calculate_mse, calculate_rmse, calculate_mape
import numpy as np
from scipy.stats import entropy 
from scipy.interpolate import make_interp_spline  

def plot_distribution_comparison(df, col_gt, col_pred, title, output_dir):
    """
    Plot distribution comparison between ground truth and prediction.

    Parameters:
    - df: DataFrame containing the data
    - col_gt: Name of the column for ground truth values
    - col_pred: Name of the column for prediction values
    - title: Title of the plot
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df, x=col_gt, bins=20, color='skyblue', label='Ground Truth', kde=True)
    sns.histplot(df, x=col_pred, bins=20, color='salmon', label='Prediction', kde=True)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()

def plot_scatter_comparison(df, col_gt, col_pred, title, output_dir):
    """
    Plot scatter comparison between ground truth and prediction.

    Parameters:
    - df: DataFrame containing the data
    - col_gt: Name of the column for ground truth values
    - col_pred: Name of the column for prediction values
    - title: Title of the plot
    """
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col_gt, y=col_pred, alpha=0.5)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()

def plot_actual_vs_prediction(df, x_col, y_actual_col, y_pred_col, title, x_label, y_label, output_dir):
    """
    Plot actual vs prediction curves for a specific dataset and columns.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - x_col (str): The column name for the x-axis (usually the index or sample index).
    - y_actual_col (str): The column name for the actual values.
    - y_pred_col (str): The column name for the predicted values.
    - title (str): The title of the plot.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    """
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df, x=x_col, y=y_actual_col, label='Ground Truth')
    sns.lineplot(data=df, x=x_col, y=y_pred_col, label='Prediction')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()

def plot_the_distribution(result_df, output_dir):
    # 绘制 TRPMILES 的预测和实际曲线
    plot_actual_vs_prediction(result_df, result_df.index, 'rconview', 'fconview',
                              'creations3_GT_vs_Pred', 'Sample Index', 'Miles', output_dir)
    # 绘制 TRVLCMIN 的预测和实际曲线
    # plot_actual_vs_prediction(result_df, result_df.index, 'trvlcmin_ground_truth', 'trvlcmin_prediction',
    #                           'TRVLCMIN_GT_vs_Pred', 'Sample Index', 'Minutes', output_dir)



def num_plot1(fconview, rconview):
    # 归一化为概率分布  
    fconview_normalized = fconview / np.sum(fconview)  
    rconview_normalized = rconview / np.sum(rconview)  

    # 计算 KL 散度  
    kl_divergence = round(entropy(fconview_normalized, rconview_normalized),3)  
    # 创建平滑曲线  
    bins = np.arange(0, 1.1, 0.2)  # 设置区间为 [0, 0.2, 0.4, 0.6, 0.8, 1.0]  
    response_counts, _ = np.histogram(fconview, bins=bins)  
    pred_counts, _ = np.histogram(rconview, bins=bins)  
    x = bins[:-1] + 0.1  
    x_smooth = np.linspace(x.min(), x.max(), 300) 

    # 使用 B样条插值生成平滑曲线  
    spl_fconview = make_interp_spline(x, response_counts, k=3) 
    spl_rconview = make_interp_spline(x, pred_counts, k=3)  

    # 生成平滑的 y 轴数据  
    fconview_smooth = spl_fconview(x_smooth)  
    rconview_smooth = spl_rconview(x_smooth)  

    # 绘制平滑曲线  
    plt.figure(figsize=(10, 5))  
    plt.plot(x_smooth, fconview_smooth, label='Response', color='blue')  
    plt.plot(x_smooth, rconview_smooth, label='Pred', color='orange')  

    # 添加标题和标签  
    plt.title(f'KL:{kl_divergence}  {cnt}')  
    plt.xlabel('Value')  
    plt.ylabel('Count')  

    # 设置 x 轴刻度  
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # 设置 x 轴刻度为具体的数值  
    plt.ylim(0, max(max(fconview_smooth), max(rconview_smooth)) * 1.1)  # 设置 y 轴范围  

    plt.legend()  
    plt.grid()  

    # 保存图形为图片文件  
    plt.savefig('11.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 格式  

    # 显示图形  
   
def calculate_kl_divergence2(list1, list2):
    # 转换为浮点数并处理NaN值
    list1 = np.array(list1, dtype=float)
    list2 = np.array(list2, dtype=float)
    
    # 去除NaN值
    list1 = list1[~np.isnan(list1)]
    list2 = list2[~np.isnan(list2)]
    
    if len(list1) == 0 or len(list2) == 0:
        raise ValueError("One of the input lists is empty after removing NaN values.")

    # 计算概率分布 (直方图)
    hist1, _ = np.histogram(list1, bins='auto', density=True)
    hist2, _ = np.histogram(list2, bins='auto', density=True)
    
    # 确保两个分布具有相同的长度
    max_length = max(len(hist1), len(hist2))
    hist1 = np.pad(hist1, (0, max_length - len(hist1)), 'constant')
    hist2 = np.pad(hist2, (0, max_length - len(hist2)), 'constant')
    
    # 避免在计算 KL 散度时出现零概率
    hist1 = np.where(hist1 == 0, 1e-10, hist1)  # 防止出现零
    hist2 = np.where(hist2 == 0, 1e-10, hist2)  # 防止出现零

    # 计算 KL 散度
    kl_divergence = entropy(hist1, hist2)
    
    return kl_divergence

def main(file_path):
    # 读取 CSV 文件
    result_df = pd.read_csv(file_path)

    # 获取所需的列数据
    # trpmiles_gt_list = result_df['trpmiles_ground_truth'].tolist()
    # trpmiles_pred_list = result_df['trpmiles_prediction'].tolist()
    # trvlcmin_gt_list = result_df['trvlcmin_ground_truth'].tolist()
    # trvlcmin_pred_list = result_df['trvlcmin_prediction'].tolist()

    # 计算指标
    # list1 = result_df['gts'].tolist()
    # list2 = result_df['llama3'].tolist()
    # list3 = result_df['llama3.1'].tolist()
    list1 = result_df['gts'].tolist()
    list2 = result_df['gpt35'].tolist()
    list3 = result_df['gpt4'].tolist()
    kl_trpmiles1 = calculate_kl_divergence2(list2, list1)
    kl_trpmiles2 = calculate_kl_divergence2(list3, list1)

    print("35 ", round(kl_trpmiles1,4))
    print("4 ", round(kl_trpmiles2,4))



if __name__ == "__main__":
    # 调用主函数并传入CSV文件路径
    # main('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round2_zero.csv')
    # main('/home/cyyuan/ACL2025/Data/RECS/duolun/KWH_round2_few.csv')
    main('/home/cyyuan/ACL2025/Data/Trell social media usage/duolungpt/conview_round1_few.csv')
    main('/home/cyyuan/ACL2025/Data/Trell social media usage/duolungpt/conview_round2_few.csv')

    
 
