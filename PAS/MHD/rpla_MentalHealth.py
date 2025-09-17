import os
import json
import plot
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import gen_request_prompt_mental
from llm_handler import LLMHandler
from metrics import calculate_accuracy, calculate_mse, calculate_rmse, calculate_mape, calculate_distribution_metrics
from utils import load_json, load_api_keys, load_data, expand_range, generate_filename, filter_and_sample_data, \
    initialize_results, create_data_to_add, parse_llm_response_mental


def main():
    parser = argparse.ArgumentParser(description="LLM-based survey prediction system")
    # Argument to specify the LLM and version
    parser.add_argument("--llm", type=str, default="llama3:8b", help="Specify the LLM and its version, e.g., 'gpt-3.5-turbo' or 'llama3.1:8b'. Default is 'gpt-3.5-turbo'.")
    # Argument to specify whether to use an API or local model
    parser.add_argument("--api", action="store_true",help="Use this flag to indicate if the LLM should be called via API.")
    parser.add_argument("--num_samples", type=int, default=29001, help="randomly selected sample size (0 for all data without sampling)")
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
    config = load_json('config/config_mental.json')

    llm_name = args.llm
    os.makedirs(os.path.join(config['result_folder'], llm_name.replace(":", "-")), exist_ok=True)

    # Display configuration
    print(f"Configuration:\n- LLM: {args.llm}\n- Using API: {args.api}\n- Num samples: {args.num_samples}\n- Group rule: {args.group_rule}\n- Enable repeated inference: {args.enable_repeated_inference}\n- Repeated inference count: {args.num_repeats}\n- Enable secondary inquiries: {args.enable_secondary_inquiries}\n")

    # Load Data
    mental_df = pd.read_csv('data/mental_health/Mental_Health_Dataset_with_index.csv')

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

    mappings = load_json(os.path.join(config['mappings_folder'], 'mapping_mental.json'))

    # init the result dict of different field lists
    id_fields = config['id_fields']
    pred_fields = config['pred_fields']
    results = initialize_results(id_fields, pred_fields)
    # print(results)

    last_houseid = 0
    losing_sample_count = 0
    lost_samples = []
    survey_background_prompt =  """This is a mental health survey sponsored by a renowned research institute, designed to understand the mental health conditions 
        and coping strategies of individuals from different backgrounds. The survey aims to explore various mental health issues, including 
        stress levels, mood swings, coping mechanisms, and social interactions. This information will help improve mental health support systems 
        and inform future mental health policies. Now, I will provide some basic profiles of the respondents:"""
    for _, mental_row in tqdm(filtered_sampled_df.iterrows(), total=filtered_sampled_df.shape[0], desc="Generating prompts and querying GPT"):
        value = {field: str(gen_request_prompt_mental.convert_value(mental_row[field])) for field in config['situation']}

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


        condq_prompt = gen_request_prompt_mental.generate_condq_prompt(value, mappings, config['cond_fields'], value['Timestamp'],config['option_field'], config['number_field'])


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

                # Reconfigure the LLMHandler for each iteration if needed
                # llm_handler.reconfigure(llm=args.llm, api_keys=api_keys, use_api=args.api)
                # break

            # predictions = repeated_inference(llm_handler, prompt, num_repeats=100)
            # # Analyze predictions and compute the gap
            # analysis_results = analyze_predictions(predictions, (trpmiles, trvlcmin))
            # results["trpmiles_prediction_mean"].append(analysis_results["trpmiles_mean"])
            # results["trpmiles_variance"].append(analysis_results["trpmiles_variance"])
            # results["trpmiles_gap"].append(analysis_results["trpmiles_gap"])
            # results["trvlcmin_prediction_mean"].append(analysis_results["trvlcmin_mean"])
            # results["trvlcmin_variance"].append(analysis_results["trvlcmin_variance"])
            # results["trvlcmin_gap"].append(analysis_results["trvlcmin_gap"])

        else:
            # Single inference for one sample
            response = llm_handler.handle_one_inquiry(prompt, args.enable_secondary_inquiries)
            # print("LLM Reply: ", response)
            # llm_handler.reconfigure(llm=args.llm, api_keys=api_keys, use_api=args.api)
            # Parsing out the predictions and counting the metrics
            try:
                pred_results = parse_llm_response_mental(response)
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


