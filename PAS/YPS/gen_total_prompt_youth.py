import os
import csv
import json
import pandas as pd
from tqdm import tqdm
from gen_request_prompt_youth import convert_value, generate_condq_prompt


def load_json(file_path):
    """Helper function to load a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_prompt():
    # Load the config files
    config = load_json('config/config_youth.json')
    mappings = load_json(os.path.join(config['mappings_folder'], 'mapping_youth.json'))

    # Load the dataset
    df = pd.read_csv('data/Young_people_survey/responses_index.csv')

    # Define the background prompt that will appear in each generated prompt
    survey_background_prompt = """You are a statistician and a social survey expert.
                I will provide you with detailed demographic and personal information about a young respondent from the youth_survey dataset, including their preferences for different music genres.
                Your task is to accurately analyze this young respondent’s profile and predict their ratings for various movie genres based on their music preferences. 
                The movie genres include, but are not limited to, Horror, Thriller, Comedy, Sci-fi, Action, Fantasy/Fairy tales, and more.
                Use the provided music preferences to determine the young respondent’s possible affinity for different types of movies.
            .\n"""


    # Define the header for the output CSV file
    header = ["UNIID", "Age", "Gender"] + config['pred_fields'] + ["PROMPT"]

    # Open the output CSV file to write the prompts
    with open("data/Young_people_survey/prompt_csv/youth_total_prompts.csv", mode='w', newline='') as file:
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
            condq_prompt = generate_condq_prompt(value, mappings, config['cond_fields'], mental_row,config['option_field'], config['number_field'])

            # Prepare the final prompt by adding the survey background prompt at the beginning
            prompt = survey_background_prompt + "\n" + condq_prompt

            # Write the data to the output CSV file
            row_data = [uniid, mental_row['Age'],mental_row['Gender']] + [
                pred_values.get(field, '') for field in config['pred_fields']] + [prompt]
            writer.writerow(row_data)


if __name__ == '__main__':
    generate_prompt()
