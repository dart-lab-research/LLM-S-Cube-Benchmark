import os
import csv
import json
from tqdm import tqdm
import gen_request_prompt_NHTS
from utils import load_json, load_data

def main():
    # Load config
    config = load_json('config/config.json')
    pred_fields = config['pred_fields']

    # Load Data
    trip_entire_df, hh_prompt_df = load_data(config)

    # Mapping of option-prompt
    with open(os.path.join(config['mappings_folder'], 'mapping.json'), 'r') as f:
        mappings = json.load(f)


    last_houseid = 0
    survey_bakground_prompt = "There is a survey, sponsored by the U.S. Department of Transportation and conducted by Ipsos Research, which selected the participant's household from across the United States to represent and understand Americans' transportation needs and experiences. The study explores transportation experiences in the participant's community and nationwide. The results will inform transportation spending decisions. Now I will provide some basic profiles of the participants."
            # save the prompt to a csv file
    with open("data/2017/csv/prompt_csv/total_prompts.csv", mode='w', newline='') as file:
        # table header
        header = ["UNIID", "HOUSEID", "PERSONID", "TRPTRANS"] + pred_fields + ["PROMPT"]
        writer = csv.writer(file)
        writer.writerow(header)
        
        for uniid, (i, trip_row) in enumerate(tqdm(trip_entire_df.iterrows(), total=trip_entire_df.shape[0], desc="Generating prompts"), start=1):
            # Convert and map values
            value = {field: str(gen_request_prompt.convert_value(trip_row[field])) for field in config['trip_fields']}
            houseid, personid, trptrans = int(float(value['HOUSEID'])), int(float(value['PERSONID'])), int(float(value['TRPTRANS']))

            # Ground Truth: Assuming pred_fields is a list containing the keys for prediction fields
            pred_values = {field.lower(): float(value[field.upper()]) for field in pred_fields}

            condq_prompt = gen_request_prompt.generate_condq_prompt(value, mappings, config['cond_fields'], personid, config['option_field'], config['number_field'])

            if houseid != last_houseid:
                hh_prompt = hh_prompt_df.loc[hh_prompt_df['HOUSEID'] == houseid, 'PROMPT'].values[0]
                prompt = survey_bakground_prompt + hh_prompt + condq_prompt
                last_houseid = houseid
            else:
                prompt = condq_prompt

            # Write to CSV with pred_values and UNIID
            row = [uniid, houseid, personid, trptrans] + [pred_values.get(field.lower(), '') for field in pred_fields] + [prompt]
            writer.writerow(row)

if __name__ == "__main__":
    main()


