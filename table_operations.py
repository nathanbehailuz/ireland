import json
import pandas as pd
from difflib import SequenceMatcher


def json_to_df(data, save = False):
    # #region agent log
    import json as json_lib
    num_parishes = len(data.get('parishes', []))
    total_entries = sum(len(p.get('entries', [])) for p in data.get('parishes', []))
    with open('/Users/nathanbehailu/Desktop/projects/ireland/.cursor/debug.log', 'a') as f: f.write(json_lib.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B,D","location":"table_operations.py:6","message":"json_to_df entry","data":{"num_parishes":num_parishes,"total_entries":total_entries},"timestamp":__import__('time').time()*1000})+'\n')
    # #endregion
    df = pd.DataFrame(columns=['parish', 'mr', 'townland', 'os', 'sublocation_1', 'sublocation_2', 'occupier', 'lessor', 'desc', 'area', 'land_val', 'building_val', 'total_val', 'n_shared', 'is_total', 'is_exemption'])
    
    for parish_data in data['parishes']:
        parish_name = parish_data['parish']
        prev_townland = None
        # #region agent log
        entries_in_parish = parish_data.get('entries', [])
        townlands_with_names = [e.get('townland', '') for e in entries_in_parish if e.get('townland', '')]
        empty_townlands = [i for i, e in enumerate(entries_in_parish) if not e.get('townland', '')]
        with open('/Users/nathanbehailu/Desktop/projects/ireland/.cursor/debug.log', 'a') as f: f.write(json_lib.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B","location":"table_operations.py:11","message":"processing parish","data":{"parish_name":parish_name,"num_entries":len(entries_in_parish),"num_with_townland":len(townlands_with_names),"num_empty_townland":len(empty_townlands),"townlands_in_parish":townlands_with_names},"timestamp":__import__('time').time()*1000})+'\n')
        # #endregion
        for entry in parish_data['entries']:
            townland = entry.get('townland', '')
            # #region agent log
            townland_before = townland
            prev_townland_before = prev_townland
            # #endregion
            if townland != "":
                prev_townland = townland
            # #region agent log
            final_townland = townland if townland != "" else prev_townland
            with open('/Users/nathanbehailu/Desktop/projects/ireland/.cursor/debug.log', 'a') as f: f.write(json_lib.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B","location":"table_operations.py:13","message":"townland filling logic","data":{"townland_in_entry":townland_before,"prev_townland":prev_townland_before,"final_townland":final_townland,"was_filled":townland_before == "" and final_townland != ""},"timestamp":__import__('time').time()*1000})+'\n')
            # #endregion
            df = df._append({
                'parish': parish_name,
                'mr': entry.get('mr', ''),
                'townland': townland if townland != "" else prev_townland,
                'os': entry.get('os', ''),
                'sublocation_1': entry.get('sublocation_1', ''),
                'sublocation_2': entry.get('sublocation_2', ''),
                'occupier': entry.get('occupier', ''),
                'lessor': entry.get('lessor', ''),
                'desc': entry.get('desc', ''),
                'area': entry.get('area', ''),
                'land_val': entry.get('land_val', ''),
                'building_val': entry.get('building_val', ''),
                'total_val': entry.get('total_val', ''),
                'n_shared': entry.get('n_shared', 0),
                'is_total': entry.get('is_total', 0),
                'is_exemption': entry.get('is_exemption', 0),
            }, ignore_index=True)
    # #region agent log
    unique_townlands_final = df['townland'].unique().tolist()
    with open('/Users/nathanbehailu/Desktop/projects/ireland/.cursor/debug.log', 'a') as f: f.write(json_lib.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B","location":"table_operations.py:34","message":"json_to_df exit","data":{"final_unique_townlands":len(unique_townlands_final),"townland_list":unique_townlands_final,"total_rows":len(df)},"timestamp":__import__('time').time()*1000})+'\n')
    # #endregion
    return df

def to_total_pence(value):
    '''
    Unit,Composition,Total Pence (d)
    1 Pound (£),20 Shillings,240d
    1 Shilling (s),12 Pence,12d
    1 Penny (d),4 Farthings,1d
    '''
    
    # Handle empty or invalid values
    if not value or value.strip() == '':
        return 0
    
    value = value.split(' ')
    
    # Ensure we have 3 parts, pad with zeros if needed
    while len(value) < 3:
        logging.error(f"Padding value with zeros: {value}")
        return 0

    pounds = int(value[0].strip())
    shillings = int(value[1].strip())
    pence = int(value[2].strip())
    
    return pounds * 240 + shillings * 12 + pence


def from_total_pence(total_pence):
    pounds = total_pence // 240
    remaining_pence = total_pence % 240
    
    shillings = remaining_pence // 12
    pence = remaining_pence % 12
    
    return (f'{pounds} {shillings} {pence}')

def check_if_correct(total_land_val, total_total_val, sum_land_val, sum_total_val):
    
    # TODO: does the check work?
    def off_by_factor_of_two(a, b):
        return a <= 2 * b and b <= 2 * a
    
    return off_by_factor_of_two(total_land_val, sum_land_val) and off_by_factor_of_two(total_total_val, sum_total_val)

def clean_townland(townland):
    return townland.upper().replace('-', '').replace(' ', '').replace('.', '')

def find_best_townland_match(expected_townland, available_townlands, threshold=0.65):
    '''
    Find the best matching townland from available options using fuzzy string matching.
    Returns (matched_townland, similarity_score) or (None, 0) if no good match found.
    '''
    best_match = None
    best_score = 0
    
    for townland in available_townlands:
        # Calculate similarity ratio
        score = SequenceMatcher(None, clean_townland(expected_townland), clean_townland(townland)).ratio()
        if score > best_score:
            best_score = score
            best_match = townland
    
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score

def get_total_values(df, townland):
    '''
    df = df[townland] -> get the df for the townland
    total_land_val, total_total_val = value of land and total for the townland at the row where is_total = 1
    '''
    df_townland = df[df['townland'] == townland]
    
    # If exact match not found, try fuzzy matching
    if len(df_townland) == 0:
        available_townlands = df['townland'].unique().tolist()
        matched_townland, score = find_best_townland_match(townland, available_townlands)
        
        if matched_townland:
            df_townland = df[df['townland'] == matched_townland]
        else:
            raise ValueError(f"Townland '{townland}' not found in extracted data (best match score: {score*100:.1f}%). Available townlands: {available_townlands}")
    
    # Check if there's a total row for this townland
    total_rows = df_townland[df_townland['is_total'] == 1]
    if len(total_rows) == 0:
        raise ValueError(f"No total row (is_total=1) found for townland '{townland}'")
    
    total_land_val = to_total_pence(total_rows['land_val'].values[0])
    total_total_val = to_total_pence(total_rows['total_val'].values[0])

    sum_land_val, sum_total_val = 0 , 0 
    for index, row in df_townland[df_townland['is_total'] == 0].iterrows():
        sum_land_val += to_total_pence(row['land_val'])
        sum_total_val += to_total_pence(row['total_val'])
    is_correct = check_if_correct(total_land_val, total_total_val, sum_land_val, sum_total_val)
    
    sum_land_val, sum_total_val = from_total_pence(sum_land_val), from_total_pence(sum_total_val)

    total_land_val, total_total_val = from_total_pence(total_land_val), from_total_pence(total_total_val)

    return {'sum_land_val': sum_land_val, 'total_land_val': total_land_val, 'sum_total_val': sum_total_val, 'total_total_val': total_total_val, 'is_correct': is_correct}
