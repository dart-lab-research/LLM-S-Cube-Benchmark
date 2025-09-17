import os
import re
import json
import pandas as pd
from pathlib import Path

# 从配置文件加载参数
def load_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_api_keys(api_key_path):
    with open(api_key_path, "r") as f:
        return json.load(f)

# 读取数据
def load_data(config):
    data_root_folder = config['data_root_folder']
    # trip_entire_df = pd.read_csv(os.path.join(data_root_folder, config['trip_entire_csv']))
    # hh_prompt_df = pd.read_csv(os.path.join(data_root_folder,config['hh_prompt_csv']))
    trip_entire_df = pd.read_csv(Path(data_root_folder) / config['mental_csv'])
    return trip_entire_df

def expand_range(value):
    if isinstance(value, str) and '-' in value:
        start, end = map(int, value.split('-'))
        return list(range(start, end + 1))
    return value

def generate_filename(filter_rules, num_samples, enable_secondary_inquiries):
    rule_parts = []
    for field, values in filter_rules.items():
        values_str = '-'.join(map(str, values)) if isinstance(values, list) else values
        rule_parts.append(f"{field}_{values_str}")
    rule_str = "_".join(rule_parts)
    
    # Add secondary inquiries status to the filename
    secondary_inquiries_str = "secondary_inquiries_enabled" if enable_secondary_inquiries else "secondary_inquiries_disabled"
    
    # Final filename
    filename = f"{rule_str}_{secondary_inquiries_str}_samples_{num_samples}.csv"
    return filename

# 过滤和采样数据
def filter_and_sample_data(save_folder_path, filter_rules, trip_entire_df, num_samples, enable_secondary_inquiries):
    # 先进行数据采样
    if num_samples == 0:
        sampled_df = trip_entire_df   # 不进行sample，所有数据
    else:
        sampled_df = trip_entire_df.sample(n=num_samples, random_state=1)
    # print(sampled_df.shape)

    if filter_rules == {}:
        # print("No filter rules applied.")
        # print(sampled_df.shape)
        return sampled_df
    # 初始化筛选后的 DataFrame
    filtered_sampled_df = sampled_df
    # print(filtered_sampled_df)

    # 根据 filter_rules 中的规则筛选数据
    for field, values in filter_rules.items():
        # print(field, values)
        # 扩展范围表达式
        expanded_values = expand_range(values) if isinstance(values, str) else values
        # print(expanded_values)
        filtered_sampled_df = filtered_sampled_df[filtered_sampled_df[field].isin(expanded_values)]
        # print(filtered_sampled_df.shape)

    filename = generate_filename(filter_rules, num_samples, enable_secondary_inquiries)
    save_path = os.path.join(save_folder_path, filename)
    filtered_sampled_df.to_csv(save_path, index=False)

    return filtered_sampled_df


def initialize_results(id_fields, pred_fields):
    # 根据传入的字段初始化结果字典。
    results = {}
    for field in id_fields:
        results[f"{field}"] = []
    for field in pred_fields:
        results[f"{field}_ground_truth"] = []
        results[f"{field}_prediction"] = []
    return results

def create_data_to_add(id_fields, pred_fields, id_values, pred_values, pred_results):
    """
    Create the data_to_add dictionary dynamically based on id_fields and pred_fields.
    
    Parameters:
    - id_fields (list): List of identifier fields (e.g., ['houseid', 'personid']).
    - pred_fields (list): List of prediction fields (e.g., ['trpmiles', 'trvlcmin']).
    - id_values (dict): Dictionary of ID field values (e.g., {'houseid': 123, 'personid': 456}).
    - pred_values (dict): Dictionary of ground truth values for prediction fields.
    - pred_results (dict): Dictionary of predicted values for prediction fields.
    
    Returns:
    - data_to_add (dict): Dictionary containing ground truth and prediction values for each field.
    """
    data_to_add = {}
    
    # Add ID fields to the dictionary
    for field in id_fields:
        data_to_add[field] = id_values.get(field)

    # Add ground truth and prediction fields to the dictionary
    for field in pred_fields:
        data_to_add[f"{field}_ground_truth"] = pred_values.get(f"{field}")
        data_to_add[f"{field}_prediction"] = pred_results.get(f"{field}_prediction")
    
    return data_to_add

# def parse_llm_response(response):
#     """
#     Parses the llm response to extract relevant predictions (e.g., distance and duration).
#
#     Parameters:
#     - response (str): The llm response text.
#
#     Returns:
#     - dict: Dictionary with parsed predictions.
#     """
#     # 默认初始化预测结果为空
#     # pred_trpmiles = None
#     # pred_trvlcmin = None
#     pred_trptrans = None
#     # 使用正则表达式查找文本中的两个数字
#     matches = re.findall(r'(\d+\.?\d*)', response)
#
#     if len(matches) >= 2:
#         # 提取前两个匹配的数字
#         pred_trpmiles = float(matches[0])  # 第一个数字作为trpmiles_prediction
#         pred_trvlcmin = float(matches[1])  # 第二个数字作为trvlcmin_prediction
#
#     # 构造预测结果字典
#     # pred_results = {
#     #     "trpmiles_prediction": pred_trpmiles,
#     #     "trvlcmin_prediction": pred_trvlcmin
#     # }
#     # pred_results = {
#     #     "trpmiles_prediction": pred_trpmiles,
#     #     "trvlcmin_prediction": pred_trvlcmin
#     # }
#     pred_results = {
#         "pred_trptrans": pred_trptrans,
#     }
#     return pred_results


import re

def parse_llm_response_mental(response):
    """
    Parses the llm response to extract relevant predictions (e.g., distance and duration).
    Now handles string responses like 'B, A, A, A' for the different fields
    and maps them to meaningful labels.

    Parameters:
    - response (str): The llm response text.

    Returns:
    - dict: Dictionary with parsed predictions mapped to meaningful labels.
    """
    # 映射字典
    stress_map = {'A': 'Yes', 'B': 'Maybe', 'C': 'No'}
    mood_map = {'A': 'High', 'B': 'Medium', 'C': 'Low'}
    coping_map = {'A': 'Yes', 'B': 'No'}
    weakness_map = {'A': 'Yes', 'B': 'Maybe', 'C': 'No'}

    # 默认初始化预测结果为空
    pred_Growing_Stress = None
    pred_Mood_Swings = None
    pred_Coping_Struggles = None
    pred_Social_Weakness = None

    # 使用正则表达式查找文本中的四个大写字母
    matches = re.findall(r'([A-Z]),\s*([A-Z]),\s*([A-Z]),\s*([A-Z])', response)

    if matches:  # 确保至少找到一组匹配
        # 提取匹配的四个字母
        Growing_Stress, Mood_Swings, Coping_Struggles, Social_Weakness = matches[0]

        # 映射每个字母到对应的含义
        pred_Growing_Stress = stress_map.get(Growing_Stress, None)
        pred_Mood_Swings = mood_map.get(Mood_Swings, None)
        pred_Coping_Struggles = coping_map.get(Coping_Struggles, None)
        pred_Social_Weakness = weakness_map.get(Social_Weakness, None)

    # 构造结果字典
    pred_results = {
        "Growing_Stress_prediction": pred_Growing_Stress,
        "Mood_Swings_prediction": pred_Mood_Swings,
        "Coping_Struggles_prediction": pred_Coping_Struggles,
        "Social_Weakness_prediction": pred_Social_Weakness
    }

    return pred_results
import re

def parse_llm_response_employee( response):
    """
    Parses the llm response to extract relevant predictions (e.g., distance and duration).
    Now handles string responses like 'B, A, A, A' for the different fields
    and maps them to meaningful labels.

    Parameters:
    - response (str): The llm response text.

    Returns:
    - dict: Dictionary with parsed predictions mapped to meaningful labels.
    """
    # 映射字典
    wlb_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    workenv_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    workload_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    stress_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    jobsatisfaction_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}

    # 默认初始化预测结果为空
    pred_WLB = None
    pred_WorkEnv = None
    pred_Workload = None
    pred_Stress = None
    pred_JobSatisfaction = None

    # 使用正则表达式查找文本中的五个大写字母
    matches = re.findall(r'([A-E]),\s*([A-E]),\s*([A-E]),\s*([A-E]),\s*([A-E])', response)

    if matches:  # 确保至少找到一组匹配
        # 提取匹配的五个字母
        WLB, WorkEnv, Workload, Stress, JobSatisfaction = matches[0]

        # 映射每个字母到对应的含义
        pred_WLB = wlb_map.get(WLB, None)
        pred_WorkEnv = workenv_map.get(WorkEnv, None)
        pred_Workload = workload_map.get(Workload, None)
        pred_Stress = stress_map.get(Stress, None)
        pred_JobSatisfaction = jobsatisfaction_map.get(JobSatisfaction, None)

    # 构造结果字典
    pred_results = {
        "WLB_prediction": pred_WLB,
        "WorkEnv_prediction": pred_WorkEnv,
        "Workload_prediction": pred_Workload,
        "Stress_prediction": pred_Stress,
        "JobSatisfaction_prediction": pred_JobSatisfaction
    }

    return pred_results

def parse_llm_response_youth(response):
    """
    Parses the llm response to extract relevant predictions (e.g., distance and duration).
    Now handles string responses like 'B, A, A, A' for the different fields
    and maps them to meaningful labels.

    Parameters:
    - response (str): The llm response text.

    Returns:
    - dict: Dictionary with parsed predictions mapped to meaningful labels.
    """
    # 映射字典
    Action_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    Documentary_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    Thriller_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    Comedy_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}

    # 默认初始化预测结果为空
    pred_Action = None
    pred_Documentary = None
    pred_Thriller = None
    pred_Comedy = None

    # 使用正则表达式查找文本中的四个大写字母
    matches = re.findall(r'([A-Z]),\s*([A-Z]),\s*([A-Z]),\s*([A-Z])', response)

    if matches:  # 确保至少找到一组匹配
        # 提取匹配的四个字母
        Action, Documentary, Thriller, Comedy = matches[0]

        # 映射每个字母到对应的含义
        pred_Action = Action_map.get(Action, None)
        pred_Documentary = Documentary_map.get(Documentary, None)
        pred_Thriller = Thriller_map.get(Thriller, None)
        pred_Comedy = Comedy_map.get(Comedy, None)

    # 构造结果字典
    pred_results = {
        "Action_prediction": pred_Action,
        "Documentary_prediction": pred_Documentary,
        "Thriller_prediction": pred_Thriller,
        "Comedy_prediction": pred_Comedy
    }

    return pred_results

def parse_llm_response_music_mental(response):
    """
    Parses the llm response to extract relevant predictions (e.g., distance and duration).
    Now handles string responses like 'B, A, A, A' for the different fields
    and maps them to meaningful labels.

    Parameters:
    - response (str): The llm response text.

    Returns:
    - dict: Dictionary with parsed predictions mapped to meaningful labels.
    """
    # 映射字典
    # Action_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    # Documentary_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    # Thriller_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    # Comedy_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}

    # 默认初始化预测结果为空
    pred_Anxiety = None
    pred_Depression = None
    pred_Insomnia = None
    pred_OCD = None

    # 使用正则表达式查找文本中的四个大写字母
    matches = re.findall(r'(\d),\s*(\d),\s*(\d),\s*(\d)', response)

    if matches:  # 确保至少找到一组匹配
        # 提取匹配的四个字母
        Anxiety, Depression, Insomnia, OCD = matches[0]

        # 映射每个字母到对应的含义
        pred_Anxiety = Anxiety
        pred_Depression = Depression
        pred_Insomnia = Insomnia
        pred_OCD = OCD

    # 构造结果字典
    pred_results = {
        "Anxiety_prediction": pred_Anxiety,
        "Depression_prediction": pred_Depression,
        "Insomnia_prediction": pred_Insomnia,
        "OCD_prediction": pred_OCD
    }

    return pred_results
def parse_llm_response_NHTS(response):
    """
    Parses the llm response to extract relevant predictions (e.g., distance and duration).

    Parameters:
    - response (str): The llm response text (e.g., '2.5, 6.6, B').

    Returns:
    - dict: Dictionary with parsed predictions.
    """
    # 默认初始化预测结果为空
    pred_trpmiles = None
    pred_trvlcmin = None
    pred_trptrans = None

    # 使用正则表达式查找两个浮动数字和一个大写字母
    matches = re.findall(r'([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+),\s*([A-Z])', response)

    if matches:  # 确保至少找到一组匹配
        # 提取第一个匹配组（假设只有一个目标句子）
        trpmiles, trvlcmin, trptrans = matches[0]

        pred_trpmiles = float(trpmiles)  # 第一个浮动数字作为 trpmiles_prediction
        pred_trvlcmin = float(trvlcmin)  # 第二个浮动数字作为 trvlcmin_prediction

        if trptrans in "ABCDEFGHIJ":  # 验证字母是否在目标范围内
            # 将字母转换为数字
            pred_trptrans = ord(trptrans) - ord('A') + 1

    # 构造结果字典
    pred_results = {
        "trpmiles_prediction": pred_trpmiles,
        "trvlcmin_prediction": pred_trvlcmin,
        "trptrans_prediction": pred_trptrans
    }

    return pred_results