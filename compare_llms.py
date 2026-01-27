import pandas as pd
import json
import os
import logging
from datetime import datetime

# Configure logging BEFORE importing other modules
# This ensures our configuration takes precedence
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'results/comparison_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
        logging.StreamHandler()
    ],
    force=True  # Override any existing logging configuration
)

# Now import modules that may configure logging
from extract import extract_table_data
from table_operations import json_to_df, get_total_values, find_best_townland_match

llms = ["claude", "openai", "gemini"]

def get_folder(image_name):
    """Extract folder number from image name (e.g., 'IRE_GRIFF_004_065.jpg' -> '004')"""
    return image_name.split('_')[2]

def get_image_path(image_name):
    """Construct full path to image file"""
    folder = get_folder(image_name)
    return f"Nanonets/analysis/{folder}/{image_name}"

def get_pages_to_check():
    with open("random_images.json", "r") as f:
        one_page_townland_images = json.load(f)
    return one_page_townland_images

def get_table(image, llm):
    image_path = get_image_path(image)
    data = extract_table_data(image_path, llm)
    if data:
        os.makedirs(f"results/{llm}", exist_ok=True)
        with open(f"results/{llm}/{image.replace('.jpg', '.json')}", "w") as f:
            json.dump(data, f)
        return data
    return None

def main(df, pages_to_check, res_df):
    pages_to_check = [page for page in pages_to_check if page not in res_df['page'].unique()]
    for page in pages_to_check:
        logging.info(f"Checking page {page}")
        for llm in llms:
            logging.info(f"using {llm}")
            table = get_table(page, llm)
            extraction_success = False
            if table:
                try:
                    df_llm = json_to_df(table)
                    # check if the num of townlands in df, is equal to the num of townlands in table
                    if len(df_llm['townland'].unique().tolist()) != len(df[df['target_filename'] == page]['townland'].unique().tolist()):
                        logging.error(f"Number of townlands in df_llm ({len(df_llm['townland'].unique().tolist())}) is not equal to the number of townlands in df ({len(df[df['target_filename'] == page]['townland'].unique().tolist())}) for {page} and {llm}")
                        logging.error(f"df_llm: {df_llm['townland'].unique().tolist()}")
                        logging.error(f"df: {df[df['target_filename'] == page]['townland'].unique().tolist()}")
                            
                    else:
                        logging.info(f"Number of townlands in df_llm is equal to the number of townlands in df")
                        extraction_success = True
                   

                    townlands_to_check = df[(df['target_filename'] == page) & (df['check_townland'] == 1)].townland.unique()
                    for townland in townlands_to_check:
                        try:
                            # Check if we need fuzzy matching for logging purposes
                            available_townlands = df_llm['townland'].unique().tolist()
                            if townland not in available_townlands:
                                matched, score = find_best_townland_match(townland, available_townlands)
                                if matched:
                                    logging.info(f"Fuzzy matched '{townland}' to '{matched}' (similarity: {score*100:.1f}%)")
                                else:
                                    logging.error(f"Failed to fuzzy match '{townland}' to any of the available townlands")
                                    logging.error(f"df_llm: {df_llm['townland'].unique().tolist()}")
                                    logging.error(f"df: {df[df['target_filename'] == page]['townland'].unique().tolist()}")
                                    continue
                            
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
                                'extraction_success': extraction_success,
                            }, ignore_index=True)
                        except ValueError as e:
                            logging.warning(f"Skipping townland '{townland}' for {page} ({llm}): {e}")
                except (KeyError, TypeError, Exception) as e:
                    logging.error(f"Failed to process table data for {page} and {llm}: {e}")
                    logging.error(f"Table structure: {type(table)} - {list(table.keys()) if isinstance(table, dict) else 'not a dict'}")
            else:
                logging.error(f"Failed to extract table for {page} and {llm}")
        
    res_df.to_csv('results/comparison_results.csv', index=False) 
    return res_df   

if __name__ == "__main__":
    delete_res = 1
    df = pd.read_excel('./nathan_to_fix.xlsx')
    pages_to_check = get_pages_to_check()
    if delete_res:
        # remove it if it exists
        if os.path.exists('results/comparison_results.csv'):    
            os.remove('results/comparison_results.csv')
        # create a new one
        res_df = pd.DataFrame(columns=['page', 'llm', 'townland', 'sum_land_val', 'total_land_val', 'sum_total_val', 'total_total_val', 'is_correct','extraction_success'])
    else:
        res_df = pd.read_csv('results/comparison_results.csv')
    
    res_df = main(df, pages_to_check, res_df)