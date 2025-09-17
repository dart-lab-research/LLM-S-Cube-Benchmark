import os
import json
import plot
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import gen_request_prompt_employee
from llm_handler import LLMHandler
from metrics import calculate_accuracy, calculate_mse, calculate_rmse, calculate_mape, calculate_distribution_metrics
from utils import load_json, load_api_keys, load_data, expand_range, generate_filename, filter_and_sample_data, \
    initialize_results, create_data_to_add,  parse_llm_response_employee


def main():
    parser = argparse.ArgumentParser(description="LLM-based survey prediction system")
    # Argument to specify the LLM and version
    parser.add_argument("--llm", type=str, default="gpt-4-turbo", help="Specify the LLM and its version, e.g., 'gpt-3.5-turbo' or 'llama3.1:8b'. Default is 'gpt-3.5-turbo'.")
    # Argument to specify whether to use an API or local model
    parser.add_argument("--api", action="store_true",default="True",help="Use this flag to indicate if the LLM should be called via API.")
    parser.add_argument("--num_samples", type=int, default=3020, help="randomly selected sample size (0 for all data without sampling)")
    parser.add_argument("--group_rule", type=str, default="filter0", help="The rule for aggregating individual predictions, e.g., 'filter1'. Default is 'filter1'.")
    parser.add_argument("--api_key_path", type=str, default="config/api_keys.json", help="Path to the API keys JSON file.")
    parser.add_argument("--enable_repeated_inference", action="store_true", help="Enable repeated inference if this flag is set. Default is False.")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of times to run repeated inference. Default is 1.")
    parser.add_argument("--enable_secondary_inquiries", action="store_true", help="Enable secondary inquiries if this flag is set. Default is False.")

    # Parse the command line arguments
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

    # Load API keys
    api_keys = load_api_keys(args.api_key_path)

    # Initialize the LLM handler
    llm_handler = LLMHandler(llm=args.llm, api_keys=api_keys, use_api=args.api)

    # Load config
    config = load_json('config/config_employee.json')

    llm_name = args.llm
    os.makedirs(os.path.join(config['result_folder'], llm_name.replace(":", "-")), exist_ok=True)

    # Display configuration
    print(f"Configuration:\n- LLM: {args.llm}\n- Using API: {args.api}\n- Num samples: {args.num_samples}\n- Group rule: {args.group_rule}\n- Enable repeated inference: {args.enable_repeated_inference}\n- Repeated inference count: {args.num_repeats}\n- Enable secondary inquiries: {args.enable_secondary_inquiries}\n")

    # Load Data
    mental_df = pd.read_csv('data/employee/employee_survey.csv')

    # Handle group rule
    try:
        # Attempt to retrieve the group rule from the config
        filter_rules = config.get('group_rules', {}).get(args.group_rule, {})
        if filter_rules is None:
            raise ValueError(f"The group rule '{args.group_rule}' is not defined in the configuration.")
        # Process the filter_rules as needed
        print(f"Applying rule: {filter_rules}")
    except KeyError as e:
        print(f"KeyError: The key {e} was not found in the configuration.")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Data sampling
    exp_folder_name = generate_filename(filter_rules, args.num_samples, args.enable_secondary_inquiries).replace(".csv", "")
    exp_folder_path = os.path.join(config['exp_folder'], llm_name.replace(":", "-"), exp_folder_name)
    os.makedirs(exp_folder_path, exist_ok=True)
    print(exp_folder_path)
    filtered_sampled_df = filter_and_sample_data(exp_folder_path, filter_rules, mental_df, args.num_samples, args.enable_secondary_inquiries)
    # exit(1)
    # print(filtered_sampled_df.shape)

    # Mapping of option-prompt
    config = load_json('config/config_employee.json')
    mappings = load_json(os.path.join(config['mappings_folder'], 'mapping_employee.json'))

    # init the result dict of different field lists
    id_fields = config['id_fields']
    pred_fields = config['pred_fields']
    results = initialize_results(id_fields, pred_fields)
    # print(results)

    last_houseid = 0
    losing_sample_count = 0
    lost_samples = []
    survey_background_prompt = """You are a statistician and a social survey expert. 
        I will provide you with detailed demographic and job-related information about an employee from the employee_survey dataset.
        Your task is to accurately analyze this employee’s profile and predict their ratings for several workplace factors based on the provided data.
    .\n"""

    inference_counter = 0  # 初始化计数器
    save_every = 100
    for _, mental_row in tqdm(filtered_sampled_df.iterrows(), total=filtered_sampled_df.shape[0], desc="Generating prompts and querying GPT"):
        value = {field: str(gen_request_prompt_employee.convert_value(mental_row[field])) for field in config['situation']}
        inference_counter += 1  # 每生成一个请求计数器加1
        # Assuming id_fields is a list containing the keys for ID fields
        id_values = {field: int(float(value[field])) for field in id_fields}

        # Ground Truth: Assuming pred_fields is a list containing the keys for prediction fields
        # pred_values = {field.lower(): float(value[field.upper()]) for field in pred_fields}
        pred_values = {
            field: value[field]
            for field in pred_fields
        }
        # print(pred_values)
        # print(value)


        condq_prompt = gen_request_prompt_employee.generate_condq_prompt(value, mappings, mental_row)

        prompt = survey_background_prompt + condq_prompt
        # the prompt message is done, select LLM model and make a request
        # print(houseid,personid)
        # print(prompt)
        # break
        if args.enable_repeated_inference:
            # Repeated inference for one sample
            print("Repeated inference for one sample")
            print(pred_values)
            for iter in range(args.num_repeats):
                response = llm_handler.handle_one_inquiry(prompt, args.enable_secondary_inquiries)
                print(f"Iteration {iter+1} response:", response)


        else:

            response = llm_handler.handle_one_inquiry(prompt, args.enable_secondary_inquiries)

            try:
                pred_results = parse_llm_response_employee(response)
                # print("Predictions:", pred_results)
                if all(value is not None for value in pred_results.values()):
                    data_to_add = create_data_to_add(id_fields, pred_fields, id_values, pred_values, pred_results)
                    for key, value in data_to_add.items():
                        results[key].append(value)
                else:
                    losing_sample_count += 1
                    print(f"lost sample {losing_sample_count}")
                    # print(f"Skipping sample {houseid} {personid} due to missing predictions.")
            except ValueError:
                pass

        if inference_counter % save_every == 0:
            print(f"[INFO] Saving intermediate results at iteration {inference_counter}")
            result_df = pd.DataFrame(results)
            result_filename = generate_filename(filter_rules, args.num_samples, args.enable_secondary_inquiries)
            result_filename_path = os.path.join(config['result_folder'], llm_name.replace(":", "-"), result_filename)
            result_df.to_csv(result_filename_path, index=False)


    # Gap analysis and saving the results
    result_df = pd.DataFrame(results)
    result_filename = generate_filename(filter_rules, args.num_samples, args.enable_secondary_inquiries)
    result_filename_path = os.path.join(config['result_folder'], llm_name.replace(":", "-"), result_filename)
    result_df.to_csv(result_filename_path, index=False)
    print(f"The result file for ground truth and prediction is save to {result_filename_path}")

    # plot_folder_name = generate_filename(filter_rules, args.num_samples, args.enable_secondary_inquiries).replace(".csv", "")
    # plot_folder_path = os.path.join(config['plot_folder'], llm_name.replace(":", "-"), plot_folder_name)
    # os.makedirs(plot_folder_path, exist_ok=True)
    # plot.main(result_filename_path, plot_folder_path) # plots is a folder for visulization files
    # print(f"{losing_sample_count}", "samples are lost due to missing predictions.")
    # print("Lost samples:", lost_samples)
if __name__ == "__main__":
    main()


