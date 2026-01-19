"""
Pseudo code:
for every page with check_townland = 1 
    for phase 1, ignore pages with unique_townlands > 1
    get the list of the images

for each image in images:
    for an llm in llms:
        table = extract_table_data(image, llm)
        save table to csv inside results/llm_name/page_id.csv

for each image in images:
    open csvs from each llm
    compare the tables
        - number of check_townland cases
        - number of check_page pages
    write comparison results to results/comparison_results.csv
"""

import pandas as pd
import json
import os
from extract import extract_table_data
import logging
from table_operations import json_to_df, get_total_values

llms = ["claude", "openai", "gemini"]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
df = pd.read_excel('./nathan_to_fix.xlsx')
res_df = pd.DataFrame(columns=['page', 'llm', 'townland', 'sum_land_val', 'total_land_val', 'sum_total_val', 'total_total_val', 'is_correct'])

def get_folder(image_name):
    """Extract folder number from image name (e.g., 'IRE_GRIFF_004_065.jpg' -> '004')"""
    return image_name.split('_')[2]

def get_image_path(image_name):
    """Construct full path to image file"""
    folder = get_folder(image_name)
    return f"Nanonets/analysis/{folder}/{image_name}"

def get_pages_to_check():
    with open("one_page_townland_images.json", "r") as f:
        one_page_townland_images = json.load(f)
    return one_page_townland_images

def get_table(image, llm):
    image_path = get_image_path(image)
    data = extract_table_data(image_path, llm)
    if data:
        os.makedirs(f"results/{llm}", exist_ok=True)
        with open(f"results/{llm}/{image}.json", "w") as f:
            json.dump(data, f)
        return data
    return None


pages_to_check = get_pages_to_check()
for page in pages_to_check:
    for llm in llms[1:]:
        table = get_table(page, llm)
        logging.info(f"Table for {page} is extracted by {llm} successfully")
        if table:
            df_llm = json_to_df(table)
            townlands_to_check = df[(df['target_filename'] == page) & (df['check_townland'] == 1)].townland.unique()
            for townland in townlands_to_check:
                res = get_total_values(df_llm, townland)
                res_df = res_df._append({
                    'page': page,
                    'llm': llm,
                    'townland': townland,
                    'sum_land_val': res['sum_land_val'],
                    'total_land_val': res['total_land_val'],
                    'sum_total_val': res['sum_total_val'],
                    'total_total_val': res['total_total_val'],
                    'is_correct': res['is_correct'],
                }, ignore_index=True)
            break
        else:
            logging.error(f"Failed to extract table for {page} and {llm}")
    
res_df.to_csv('results/comparison_results.csv', index=False)


