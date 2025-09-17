import os
import json
import pandas as pd
from openai import OpenAI
import gen_request_prompt
from tqdm import tqdm
from metrics import calculate_accuracy, calculate_mse, calculate_rmse, calculate_mape, calculate_distribution_metrics

# 从配置文件加载参数
def load_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

# 初始化GPT-3.5-turbo 模型
def ask_gpt(client, messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content

def prompt_polishing(client, prompt):
    polish_example = "For a white, 68 years old female person who lives in a nonurban area in the region of Riverside-San Bernardino-Ontario, CA, in a 2-person family with an income of $50,000 to $74,999, who owns a home and has 3 cars, with some college or associates degree, with no work, ..."
    polish_request = [
        {"role": "user", "content": polish_example + "\n"}
    ]
    polish_request.append({"role": "user", "content": "Here is the prompt to be embellished: " + prompt})
    polished_prompt = ask_gpt(client, polish_request) + "\n"
    return polished_prompt


# 生成条件信息和问题
def generate_condq_prompt(value, mappings, cond_fields, personid, option_field, number_field):
    cond_info_prompt = f"As for the household member whose PersonID is {personid}, given the following conditions of this person's trips: "
    cond_info_prompt += gen_request_prompt.gen_prompt_from_fields(value, mappings, cond_fields, option_field, number_field)
    question_prompt = "Now that you are role-playing this person based on the above information. Under these conditions, for all your trips that take this travel mode on this day, please try your best to predict the total trip distance in miles ranging from 0 to 9621.053 and the total trip duration in minutes ranging from 0 to 1200. "
    prompt_format = "Your response should consist of just two numbers separated by a single comma, without any additional text or explanation.\n"
    return cond_info_prompt + question_prompt + prompt_format + "\n"

# 读取数据
def load_data(config):
    data_root_folder = config['data_root_folder']
    trip_entire_df = pd.read_csv(os.path.join(data_root_folder, config['trip_entire_csv']))
    hh_prompt_df = pd.read_csv(os.path.join(data_root_folder,config['hh_prompt_csv']))
    return trip_entire_df, hh_prompt_df


def expand_range(value):
    if isinstance(value, str) and '-' in value:
        start, end = map(int, value.split('-'))
        return list(range(start, end + 1))
    return value

def generate_filename(prefix_str, filter_rules, num_samples):
    rule_parts = []
    for field, values in filter_rules.items():
        values_str = '-'.join(map(str, values)) if isinstance(values, list) else values
        rule_parts.append(f"{field}_{values_str}")
    rule_str = "_".join(rule_parts)
    filename = f"{prefix_str}_{rule_str}_samples_{num_samples}.csv"
    return filename

# 过滤和采样数据
def filter_and_sample_data(save_folder_path, filter_rules, trip_entire_df, num_samples):
    # 先进行数据采样
    # num_samples = trip_entire_df.shape[0]   # 不进行sample，所有数据
    sampled_df = trip_entire_df.sample(n=num_samples, random_state=1)
    # print(sampled_df.shape)

    # 初始化筛选后的 DataFrame
    filtered_sampled_df = sampled_df

    # 根据 filter_rules 中的规则筛选数据
    for field, values in filter_rules.items():
        # print(field, values)
        # 扩展范围表达式
        expanded_values = expand_range(values) if isinstance(values, str) else values
        # print(expanded_values)
        filtered_sampled_df = filtered_sampled_df[filtered_sampled_df[field].isin(expanded_values)]
        # print(filtered_sampled_df.shape)

    filename = generate_filename("exp_data", filter_rules, num_samples)
    save_path = os.path.join(save_folder_path, filename)
    filtered_sampled_df.to_csv(save_path, index=False)
    
    return filtered_sampled_df

# 主程序 is Here!
def main():
    config = load_json('config/config.json')
    trip_entire_df, hh_prompt_df = load_data(config)

    # 数据采样、分组和筛选
    filter_rules = config.get('group_rules', {}).get('filter7', {})
    print(filter_rules)
    filtered_sampled_df = filter_and_sample_data(config['exp_folder'], filter_rules, trip_entire_df, config['num_samples'])


    with open(os.path.join(config['mappings_folder'], 'mapping.json'), 'r') as f:
        mappings = json.load(f)

    results = {
        "houseid": [],
        "personid": [],
        "trpmiles_ground_truth": [],
        "trpmiles_prediction": [],
        "trvlcmin_ground_truth": [],
        "trvlcmin_prediction": []
    }

    last_houseid = 0
    survey_bakground_prompt = "There is a survey, sponsored by the U.S. Department of Transportation and conducted by Ipsos Research, which selected the participant's household from across the United States to represent and understand Americans' transportation needs and experiences. The study explores transportation experiences in the participant's community and nationwide. The results will inform transportation spending decisions. Now I will provide some basic profiles of the participants.\n"
    for _, trip_row in tqdm(filtered_sampled_df.iterrows(), total=filtered_sampled_df.shape[0], desc="Generating prompts and querying GPT"):
        # print(trip_row)
        value = {field: str(gen_request_prompt.convert_value(trip_row[field])) for field in config['trip_fields']}
        houseid, personid = int(float(value['HOUSEID'])), int(float(value['PERSONID']))
        trpmiles, trvlcmin = float(value['TRPMILES']), float(value['TRVLCMIN'])

        condq_prompt = generate_condq_prompt(value, mappings, config['cond_fields'], personid, config['option_field'], config['number_field'])
        conversation_history = []

        with open(config['api_key_path'], "r") as api_key_file:
            client = OpenAI(api_key=api_key_file.read())

         # 同一个家庭用同一个gpt会话，而且用同一个家庭-成员背景提示词
        if houseid != last_houseid:
            hh_prompt = hh_prompt_df.loc[hh_prompt_df['HOUSEID'] == houseid, 'PROMPT'].values[0]
            polished_prompt = prompt_polishing(client, hh_prompt)
            # print(hh_prompt)
            # print("__"*50)
            # print(polished_prompt)
            prompt =  survey_bakground_prompt + polished_prompt + condq_prompt
            conversation_history = [{"role": "user", "content": prompt}]
        else:
            prompt = condq_prompt
            conversation_history.append({"role": "user", "content": condq_prompt})

        gpt_response = ask_gpt(client, conversation_history)
        conversation_history.append({"role": "assistant", "content": gpt_response})
        conversation_history.append({"role": "user", "content": "So, your answer is: "})
        # print(gpt_response)
        gpt_result = ask_gpt(client, conversation_history)
        # print(gpt_result)

        try:
            pred_trpmiles, pred_trvlcmin = map(float, gpt_result.split(','))
            data_to_add = {
                "houseid": houseid,
                "personid": personid,
                "trpmiles_ground_truth": trpmiles,
                "trpmiles_prediction": pred_trpmiles,
                "trvlcmin_ground_truth": trvlcmin,
                "trvlcmin_prediction": pred_trvlcmin
            }
            for key, value in data_to_add.items():
                results[key].append(value)
        except ValueError:
            pass

        last_houseid = houseid
        # break
    result_df = pd.DataFrame(results)
    result_filename = generate_filename("result", filter_rules, config['num_samples'])
    result_filename_path = os.path.join(config['result_folder'], result_filename)
    result_df.to_csv(result_filename_path, index=False)
    print(f"The result file for ground truth and prediction is save to {result_filename_path}")

    import plot
    plot.main(result_filename_path)

if __name__ == "__main__":
    main()


