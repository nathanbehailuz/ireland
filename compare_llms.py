"""
Pseudo code:
for pages with check_page = 1:
    get pages with check_townland = 1 
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

def get_pages_to_check():
    # Placeholder function to get pages with check_page = 1
    pass


