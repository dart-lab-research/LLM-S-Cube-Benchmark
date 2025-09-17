import numpy as np
from scipy.stats import entropy

def calculate_accuracy(gt_list, pred_list):
    if len(gt_list) != len(pred_list):
        raise ValueError("Lists must have the same length")

    correct_predictions = sum(1 for gt, pred in zip(gt_list, pred_list) if gt == pred)
    accuracy = correct_predictions / len(gt_list)
    return accuracy

def calculate_mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("Lists must have the same length")
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("Lists must have the same length")
    # if np.any(y_true == 0):
    #     raise ValueError("y_true contains zero values, which would cause division by zero")
    # Avoid division by zero by replacing zeros with a small value
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 计算分布指标的函数
def calculate_distribution_metrics(data_list):
    return {
        'mean': np.mean(data_list),
        'std_dev': np.std(data_list),
        'median': np.median(data_list),
        'min': np.min(data_list),
        'max': np.max(data_list)
    }

# 计算直方图来近似概率分布
def calculate_histogram(data_list, bins=10):
    # print(data_list.shape)
    hist, bin_edges = np.histogram(data_list, bins=bins, density=True)
    return hist + 1e-10, bin_edges  # 加上一个小的数值防止出现0

# 计算KL散度的函数

def calculate_kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two distributions.

    Parameters:
    - p: The true distribution (list or np.array)
    - q: The predicted distribution (list or np.array)

    Returns:
    - kl_div: The KL divergence value
    """
    p, q = np.array(p), np.array(q)
    # print(p.shape)
    # print(q.shape)
    
    # Ensure both distributions are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Avoid division by zero and log of zero
    epsilon = 1e-10
    p = np.where(p == 0, epsilon, p)
    q = np.where(q == 0, epsilon, q)
   
    # scipy计算函数可以处理非归一化情况，直接使用entropy即可
    print(entropy(p,q))
    return entropy(p, q)


def analyze_predictions(predictions, ground_truth):
    """
    分析预测结果，计算均值、方差和与真实值的差距。
    
    Args:
        predictions (list of tuples): 每个元组包含 (pred_trpmiles, pred_trvlcmin)。
        ground_truth (tuple): 真实值 (trpmiles, trvlcmin)。
    
    Returns:
        dict: 包含统计结果的字典。
    """
    pred_trpmiles_list, pred_trvlcmin_list = zip(*predictions)
    
    # 计算统计量
    trpmiles_mean = np.mean(pred_trpmiles_list)
    trpmiles_variance = np.var(pred_trpmiles_list)
    
    trvlcmin_mean = np.mean(pred_trvlcmin_list)
    trvlcmin_variance = np.var(pred_trvlcmin_list)
    
    # 计算与真实值的差距
    trpmiles_gap = trpmiles_mean - ground_truth[0]
    trvlcmin_gap = trvlcmin_mean - ground_truth[1]
    
    return {
        "trpmiles_mean": trpmiles_mean,
        "trpmiles_variance": trpmiles_variance,
        "trpmiles_gap": trpmiles_gap,
        "trvlcmin_mean": trvlcmin_mean,
        "trvlcmin_variance": trvlcmin_variance,
        "trvlcmin_gap": trvlcmin_gap
    }