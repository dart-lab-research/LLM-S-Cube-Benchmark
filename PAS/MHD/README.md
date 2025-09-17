# Usage
## 0. Data preparation
### 1. Download the 2017 data from [National Household Travel Survey](https://nhts.ornl.gov/).
### 2. Put the four csv files (hhpub.csv, perpub.csv, trippub.csv and vehpub.csv) in `data/2017/csv/origin_csv`.

## 1. Environment setting
`conda env create -f environment.yml -n gpt`  
`conda activate gpt`  
(python 3.12.1, pandas, tqdm, openai, scipy, seaborn)

## 2. Set the OpenAI API Key File
`echo "YOUR_API_KEY_HERE" > config/api_key.txt`  

## 3. Data filtering, cleaning and aggregating
`python script/data_processing.py`  
Output:
1. col_filtered_csv/
   a. hhpub_filtered.csv  
   b. perpub_filtered.csv  
   c. trippub_filtered.csv  
2. cleaned_csv/
   a. hhpub_filtered_cleaned.csv  
   b. perpub_filtered_cleaned.csv  
   c. trippub_filtered_cleaned.csv  
   d. trip_mode_remapped.csv  
   e. trip_class_remapped_aggregated.csv  

## 4. Get the entire csv for trips
`python script/get_entire_csv.py`  
Output: trip_entire_csv.csv

## 5. Generating and saving the prompts for household basic info
`python script/gen_request_prompt.py`    
Output: prompt_csv/household_info.csv  

## 6. Feed to GPT and save results
`python script/rpla.py`  

Output:
1. exp_data/exp_data_[your_exp_config]_.csv  
2. result/result_[your_exp_config]_.csv  

## 7. Proactively calculate metrics and visualize the gap
`python script/plot.py [your_result_csv_path]`  
e.g., `python script/plot.py result/result_R_AGE_60-92_samples_1000.csv`


