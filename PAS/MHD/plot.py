import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_json
from metrics import calculate_distribution_metrics, calculate_kl_divergence, calculate_mse, calculate_rmse, calculate_mape
from scipy.stats import wasserstein_distance

def analyze_distributions(df, col_gt, col_pred):
    # Ground Truth statistics
    gt_mean = df[col_gt].mean()
    gt_median = df[col_gt].median()
    gt_std = df[col_gt].std()
    
    # Prediction statistics
    pred_mean = df[col_pred].mean()
    pred_median = df[col_pred].median()
    pred_std = df[col_pred].std()
    
    # Wasserstein distance (Earth Mover's Distance)
    wasserstein_dist = wasserstein_distance(df[col_gt], df[col_pred])

    print(f"Statistics for {col_gt} (Ground Truth):")
    print(f"Mean: {gt_mean}, Median: {gt_median}, Std: {gt_std}")
    print(f"Statistics for {col_pred} (Prediction):")
    print(f"Mean: {pred_mean}, Median: {pred_median}, Std: {pred_std}")
    print(f"Wasserstein Distance: {wasserstein_dist}")
    
    return {
        'gt_mean': gt_mean,
        'gt_median': gt_median,
        'gt_std': gt_std,
        'pred_mean': pred_mean,
        'pred_median': pred_median,
        'pred_std': pred_std,
        'wasserstein_dist': wasserstein_dist
    }

def extract_columns_and_compute_metrics(df, pred_fields):
    metrics_dict = {}
    for field in pred_fields:
        gt_col = f'{field.lower()}_ground_truth'
        pred_col = f'{field.lower()}_prediction'
        original_count = len(df)
        df = df.dropna(subset=[gt_col, pred_col])
        cleaned_count = len(df)
        print(f"Discarded rows with NaN for {field}:", original_count - cleaned_count)
        # # 计算阈值
        # lower_bound = df[gt_col].quantile(0.01)
        # upper_bound = df[gt_col].quantile(0.99)

        # # 去除极端值
        # df = df[(df[gt_col] >= lower_bound) & (df[gt_col] <= upper_bound)]
        
        gt_list = df[gt_col].tolist()
        pred_list = df[pred_col].tolist()
        # print(gt_list, pred_list)

        gt_metrics = calculate_distribution_metrics(gt_list)
        pred_metrics = calculate_distribution_metrics(pred_list)

        kl_divergence = calculate_kl_divergence(pred_list, gt_list)
        mse = calculate_mse(gt_list, pred_list)
        rmse = calculate_rmse(gt_list, pred_list)
        mape = calculate_mape(gt_list, pred_list)

        metrics_dict[field] = {
            'gt_metrics': gt_metrics,
            'pred_metrics': pred_metrics,
            'kl_divergence': kl_divergence,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
    
    return metrics_dict

# def plot_distribution_comparison(df, col_gt, col_pred, title, output_dir):
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df, x=col_gt, bins=20, color='skyblue', label='Ground Truth', kde=False)
#     sns.histplot(df, x=col_pred, bins=20, color='salmon', label='Prediction', kde=False)
#     plt.title(title)
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, f"{title}.png"))
#     plt.close()

def plot_distribution_comparison(df, col_gt, col_pred, title, output_dir):
    plt.figure(figsize=(6, 4))

    # 绘制 Ground Truth 的分布曲线
    sns.kdeplot(df[col_gt], color='skyblue', label='Ground Truth', linewidth=2)

    # 绘制 Prediction 的分布曲线
    sns.kdeplot(df[col_pred], color='salmon', label='Prediction', linewidth=2)

    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()



def plot_scatter_comparison(df, col_gt, col_pred, title, output_dir):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col_gt, y=col_pred, alpha=0.5)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()

def plot_actual_vs_prediction(df, x_col, y_actual_col, y_pred_col, title, x_label, y_label, output_dir):
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

def plot_all_comparisons(df, pred_fields, output_dir):
    for field in pred_fields:
        gt_col = f'{field.lower()}_ground_truth'
        pred_col = f'{field.lower()}_prediction'
        
        plot_distribution_comparison(df, gt_col, pred_col, f'Distribution_{field.upper()}', output_dir)
        analyze_distributions(df, 'trpmiles_ground_truth', 'trpmiles_prediction')
        plot_scatter_comparison(df, gt_col, pred_col, f'Scatter_{field.upper()}', output_dir)
        plot_actual_vs_prediction(df, df.index, gt_col, pred_col, f'{field.upper()}_GT_vs_Pred', 'Sample Index', field.capitalize(), output_dir)

def save_metrics_to_file(metrics_dict, output_dir):
    with open(os.path.join(output_dir, 'metrics_results.txt'), 'w') as f:
        for field, metrics in metrics_dict.items():
            f.write(f"Metrics for {field.upper()}:\n")
            f.write(f"Ground Truth Metrics: {metrics['gt_metrics']}\n")
            f.write(f"Prediction Metrics: {metrics['pred_metrics']}\n")
            f.write(f"KL Divergence: {metrics['kl_divergence']}\n")
            f.write(f"MSE: {metrics['mse']}\n")
            f.write(f"RMSE: {metrics['rmse']}\n")
            f.write(f"MAPE: {metrics['mape']}\n\n")


def main(file_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file into a DataFrame
    result_df = pd.read_csv(file_path)

    # Load configuration settings
    config = load_json('config/config.json')
    pred_fields = config['pred_fields']  # Dynamic retrieval of prediction fields

    # Extract columns and compute metrics
    metrics_dict = extract_columns_and_compute_metrics(result_df, pred_fields)
    
    # Plot comparisons and save figures
    plot_all_comparisons(result_df, pred_fields, output_dir)
    
    # Save the calculated metrics to a file
    save_metrics_to_file(metrics_dict, output_dir)


if __name__ == "__main__":
    # Command-line interface for running the script
    if len(sys.argv) != 3:
        print("Usage: python plot.py <csv_file_path> <output_folder_path>")
    else:
        csv_file_path = sys.argv[1]
        output_folder_path = sys.argv[2]
        main(csv_file_path, output_folder_path)
