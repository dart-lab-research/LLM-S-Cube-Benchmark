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
    plot_actual_vs_prediction(result_df, result_df.index, 'gts', 'responses',
                              'support_GT_vs_Pred', 'Sample Index', 'Miles', output_dir)
    # 绘制 TRVLCMIN 的预测和实际曲线
    # plot_actual_vs_prediction(result_df, result_df.index, 'trvlcmin_ground_truth', 'trvlcmin_prediction',
    #                           'TRVLCMIN_GT_vs_Pred', 'Sample Index', 'Minutes', output_dir)



def num_plot1(responses, preds):
    # 归一化为概率分布  
    responses_normalized = responses / np.sum(responses)  
    preds_normalized = preds / np.sum(preds)  

    # 计算 KL 散度  
    kl_divergence = round(entropy(responses_normalized, preds_normalized),3)  
    # 创建平滑曲线  
    bins = np.arange(0, 1.1, 0.2)  # 设置区间为 [0, 0.2, 0.4, 0.6, 0.8, 1.0]  
    response_counts, _ = np.histogram(responses, bins=bins)  
    pred_counts, _ = np.histogram(preds, bins=bins)  
    x = bins[:-1] + 0.1  
    x_smooth = np.linspace(x.min(), x.max(), 300) 

    # 使用 B样条插值生成平滑曲线  
    spl_responses = make_interp_spline(x, response_counts, k=3) 
    spl_preds = make_interp_spline(x, pred_counts, k=3)  

    # 生成平滑的 y 轴数据  
    responses_smooth = spl_responses(x_smooth)  
    preds_smooth = spl_preds(x_smooth)  

    # 绘制平滑曲线  
    plt.figure(figsize=(10, 5))  
    plt.plot(x_smooth, responses_smooth, label='Response', color='blue')  
    plt.plot(x_smooth, preds_smooth, label='Pred', color='orange')  

    # 添加标题和标签  
    plt.title(f'KL:{kl_divergence}  {cnt}')  
    plt.xlabel('Value')  
    plt.ylabel('Count')  

    # 设置 x 轴刻度  
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # 设置 x 轴刻度为具体的数值  
    plt.ylim(0, max(max(responses_smooth), max(preds_smooth)) * 1.1)  # 设置 y 轴范围  

    plt.legend()  
    plt.grid()  

    # 保存图形为图片文件  
    plt.savefig('11.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 格式  

    # 显示图形  
   


def main(file_path, output_dir):
    # 读取 CSV 文件
    os.makedirs(output_dir, exist_ok=True)
    result_df = pd.read_csv(file_path)

    # 获取所需的列数据
    # trpmiles_gt_list = result_df['trpmiles_ground_truth'].tolist()
    # trpmiles_pred_list = result_df['trpmiles_prediction'].tolist()
    # trvlcmin_gt_list = result_df['trvlcmin_ground_truth'].tolist()
    # trvlcmin_pred_list = result_df['trvlcmin_prediction'].tolist()

    # 计算指标
    list1 = result_df['gts'].tolist()
    list2 = result_df['responses'].tolist()
    trpmiles_gt_metrics = calculate_distribution_metrics(list1)
    #trvlcmin_gt_metrics = calculate_distribution_metrics(trvlcmin_gt_list)
    trpmiles_pred_metrics = calculate_distribution_metrics(list2)
    #trvlcmin_pred_metrics = calculate_distribution_metrics(trvlcmin_pred_list)

    # 打印结果
    print("weekends_gt_metrics:", trpmiles_gt_metrics)
    #print("trvlcmin_gt_metrics:", trvlcmin_gt_metrics)
    print("trpmiles_pred_metrics:", trpmiles_pred_metrics)
    #print("trvlcmin_pred_metrics:", trvlcmin_pred_metrics)

    # 绘制 trpmiles 的分布对比图
    plot_distribution_comparison(result_df, 'gts', 'responses', 'Distribution_support_6k', output_dir)

    # # 绘制 trpmiles 的散点对比图
    plot_scatter_comparison(result_df, 'gts', 'responses', 'Scatter_support_6k', output_dir)

    # 绘制 trpmiles 和 trvlcmin 的实际和预测曲线
    plot_the_distribution(result_df, output_dir)

    # 查看两个分布的KL散度
    kl_trpmiles = calculate_kl_divergence(list1, list2)
    #kl_trvlcmin = calculate_kl_divergence(trvlcmin_pred_list, trvlcmin_gt_list)
    print("KL divergence for TRPMILES:", kl_trpmiles)
    #print("KL divergence for TRVLCMIN:", kl_trvlcmin)
    
    # 计算MSE，RMSE，MAPE
    trpmiles_mse = calculate_mse(list2, list1)
    trpmiles_rmse = calculate_rmse(list2, list1)
    trpmiles_mape = calculate_mape(list2, list1)

    # trvlcmin_mse = calculate_mse(trvlcmin_gt_list, trvlcmin_pred_list)
    # trvlcmin_rmse = calculate_rmse(trvlcmin_gt_list, trvlcmin_pred_list)
    # trvlcmin_mape = calculate_mape(trvlcmin_gt_list, trvlcmin_pred_list)

    print("MSE: " + str(trpmiles_mse))
    print("RMSE: " + str(trpmiles_rmse))
    print("MAPE: " + str(trpmiles_mape))

if __name__ == "__main__":
    # 调用主函数并传入CSV文件路径
    folder_path = './Data/Income/'
    input_path = r"/home/cyyuan/Data/Income/gpt/zero35num.csv"
    output_path = folder_path + 'output/'

    fig_path = folder_path + 'pic/'
    main(input_path, output_path)
