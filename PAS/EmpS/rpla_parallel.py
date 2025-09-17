import os
import pandas as pd
from tqdm import tqdm
from llm_handler import LLMHandler
from utils import load_api_keys, initialize_results, create_data_to_add, parse_llm_response
import multiprocessing
import time
import sys
import plot
from functools import partial

def query_ollama(prompt_with_data, model_name):
    prompt, uniid, houseid, personid, trptrans, trpmiles, trvlcmin = prompt_with_data
    api_keys = load_api_keys("config/api_keys.json")
    llm_handler = LLMHandler(llm=model_name, api_keys=api_keys, use_api=False)
    
    response = llm_handler.handle_one_inquiry(prompt, enable_secondary_inquiries=True)
    pred_results = parse_llm_response(response)
    return uniid, houseid, personid, trptrans, trpmiles, trvlcmin, pred_results

def handle_prompts_in_parallel(prompts, model_name, num_processes=None):
    if num_processes is None:
        # num_processes = int(max(1, multiprocessing.cpu_count())/4)
        num_processes = 1
        print(f"Using {num_processes} processes.")

    pool = multiprocessing.Pool(processes=num_processes)
    
    # 使用偏函数打包 model_name
    query_with_model = partial(query_ollama, model_name=model_name)

    output = []
    with tqdm(total=len(prompts)) as pbar:
        for result in pool.imap_unordered(query_with_model, prompts):
            output.append(result)
            pbar.update(1)

    pool.close()
    pool.join()

    return output

def main(model_name):
    os.environ["OLLAMA_NUM_PARALLEL"] = str(multiprocessing.cpu_count())

    prompt_file_path = "data/2017/csv/prompt_csv/total_prompts.csv"
    if not os.path.exists(prompt_file_path):
        print(f"Prompt file not found at: {prompt_file_path}")
        return

    reader = pd.read_csv(prompt_file_path)
    
    # 只取前10行数据
    prompts = reader.apply(lambda row: (row['PROMPT'], row['UNIID'], row['HOUSEID'], row['PERSONID'], row['TRPTRANS'], row['TRPMILES'], row['TRVLCMIN']), axis=1).tolist()

    # model_name = "llama3.1:8b"
    model_results = handle_prompts_in_parallel(prompts, model_name)

    # Initialize results dictionary
    results = initialize_results(['UNIID', 'HOUSEID', 'PERSONID', 'TRPTRANS'], ['TRPMILES', 'TRVLCMIN'])

    # Collect results
    for result in model_results:
        # uniid, houseid, personid, trptrans, trpmiles, trvlcmin, pred_results
        uniid, houseid, personid, trptrans, trpmiles_ground_truth, trvlcmin_ground_truth, pred_results = result

        # Create the data_to_add with correct values
        data_to_add = create_data_to_add(
            ['UNIID', 'HOUSEID', 'PERSONID', 'TRPTRANS'],
            ['TRPMILES', 'TRVLCMIN'],
            {'uniid': uniid, 'houseid': houseid, 'personid': personid, 'trptrans': trptrans},
            {'trpmiles': trpmiles_ground_truth, 'trvlcmin': trvlcmin_ground_truth},
            pred_results
        )
        for key, value in data_to_add.items():
            results[key].append(value)

    # Convert results to DataFrame
    result_df = pd.DataFrame(results).sort_values(by='uniid')
    result_filename_path = "result/"+ model_name.replace(":", "-") + "/all_samples_secondary_inquiries.csv"
    result_df.to_csv(result_filename_path, index=False)
    print(f"The result file for ground truth and prediction is saved to {result_filename_path}")

    plot_folder_path = "plots/"+ model_name.replace(":", "-") + "/all_samples_secondery_inquiries"
    plot.main(result_filename_path, plot_folder_path) # plots is a 

    

if __name__ == "__main__":
    start = time.time()
    # Command-line interface for running the script
    # print(len(sys.argv))
    if len(sys.argv) != 2:
        print("Usage: python parallel.py [model_name]")
    else:
        
        model_name = sys.argv[1]
        print(model_name)
        main(model_name)
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds.")
