import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from gen_request_prompt import convert_value, generate_condq_prompt  # 导入你给定的函数


def load_json(file_path):
    """Helper function to load a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_prompt():
    # Load the config files
    config = load_json('config/config_employee.json.json')
    mappings = load_json(os.path.join(config['mappings_folder'], 'mapping_mental.json'))

    # Load the dataset
    df = pd.read_csv('data/employee/employee_survey.csv')

    # Define the background prompt that will appear in each generated prompt
    survey_background_prompt = (
        "This is a mental health survey sponsored by a renowned research institute, designed to understand the mental health conditions "
        "and coping strategies of individuals from different backgrounds. The survey aims to explore various mental health issues, including "
        "stress levels, mood swings, coping mechanisms, and social interactions. This information will help improve mental health support systems "
        "and inform future mental health policies. Now, I will provide some basic profiles of the respondents:"
    )

    # Define the header for the output CSV file
    header = ["UNIID", "Timestamp", "Gender", "Occupation", "Country", "self_employed", "family_history", "treatment",
              "Days_Indoors"] + config['pred_fields'] + ["PROMPT"]

    # Open the output CSV file to write the prompts
    with open("data/mental_health/prompt_csv/mental_total_prompts.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # Iterate over each row in the dataset to generate prompts
        for uniid, row in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="Generating prompts"), start=1):
            mental_row = row[1]  # This will get the row as a Series
            value = {field: str(convert_value(mental_row[field])) for field in
                     config['mental_situation']}  # You may need to adjust this part based on your fields

            # Get pred_values based on the config fields
            pred_values = {field: mental_row[field] for field in config['pred_fields']}

            # Generate the conditional question prompt (assuming this function exists in gen_request_prompt)
            condq_prompt = generate_condq_prompt(value, mappings, config['cond_fields'], mental_row['Timestamp'],
                                                 config['option_field'], config['number_field'])

            # Prepare the final prompt by adding the survey background prompt at the beginning
            prompt = survey_background_prompt + "\n" + condq_prompt

            # Write the data to the output CSV file
            row_data = [uniid, mental_row['Timestamp'], mental_row['Gender'], mental_row['Occupation'], mental_row['Country'],
                        mental_row['self_employed'],
                        mental_row['family_history'], mental_row['treatment'], mental_row['Days_Indoors']] + [
                           pred_values.get(field, '') for field in config['pred_fields']] + [prompt]
            writer.writerow(row_data)


if __name__ == '__main__':
    generate_prompt()
