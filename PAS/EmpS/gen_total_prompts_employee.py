import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from gen_request_prompt_employee import convert_value, generate_condq_prompt  # 导入你给定的函数


def load_json(file_path):
    """Helper function to load a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_prompt():
    # Load the config files
    config = load_json('config/config_employee.json')
    mappings = load_json(os.path.join(config['mappings_folder'], 'mapping_employee.json'))

    # Load the dataset
    df = pd.read_csv('data/employee/employee_survey.csv')

    # Define the background prompt that will appear in each generated prompt
    survey_background_prompt = ("""
        You are a statistician and a social survey expert. 
        I will provide you with detailed demographic and job-related information about an employee from the employee_survey dataset.
        Your task is to accurately analyze this employee’s profile and predict their ratings for several workplace factors based on the provided data.
    
    """
    )

    # Define the header for the output CSV file
    header = ["UNIID", "EmpID", "Gender"] + config['pred_fields'] + ["PROMPT"]

    # Open the output CSV file to write the prompts
    with open("data/employee/prompt_csv/employee_total_prompts.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # Iterate over each row in the dataset to generate prompts
        for uniid, row in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="Generating prompts"), start=1):
            mental_row = row[1]  # This will get the row as a Series
            value = {field: str(convert_value(mental_row[field])) for field in
                     config['situation']}  # You may need to adjust this part based on your fields

            # Get pred_values based on the config fields
            pred_values = {field: mental_row[field] for field in config['pred_fields']}

            # Generate the conditional question prompt (assuming this function exists in gen_request_prompt)
            condq_prompt = generate_condq_prompt(value, mappings, mental_row)

            # Prepare the final prompt by adding the survey background prompt at the beginning
            prompt = survey_background_prompt + "\n" + condq_prompt

            # Write the data to the output CSV file
            row_data = [uniid, mental_row['EmpID']] + [
                           pred_values.get(field, '') for field in config['pred_fields']] + [prompt]
            writer.writerow(row_data)


if __name__ == '__main__':
    generate_prompt()
