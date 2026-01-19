import json
import pandas as pd


def json_to_df(data, save = False):
    df = pd.DataFrame(columns=['parish', 'mr', 'townland', 'os', 'sublocation_1', 'sublocation_2', 'occupier', 'lessor', 'desc', 'area', 'land_val', 'building_val', 'total_val', 'n_shared', 'is_total', 'is_exemption'])
    
    for parish_data in data['parishes']:
        parish_name = parish_data['parish']
        prev_townland = None
        for entry in parish_data['entries']:
            if entry['townland'] != "":
                prev_townland = entry['townland']
            df = df._append({
                'parish': parish_name,
                'mr': entry['mr'],
                'townland': entry['townland'] if entry['townland'] != "" else prev_townland,
                'os': entry['os'],
                'sublocation_1': entry['sublocation_1'],
                'sublocation_2': entry['sublocation_2'],
                'occupier': entry['occupier'],
                'lessor': entry['lessor'],
                'desc': entry['desc'],
                'area': entry['area'],
                'land_val': entry['land_val'],
                'building_val': entry['building_val'],
                'total_val': entry['total_val'],
                'n_shared': entry['n_shared'],
                'is_total': entry['is_total'],
                'is_exemption': entry['is_exemption'],
            }, ignore_index=True)
    # if save:
        # df.to_csv(json_file.replace('.json', '.csv'), index=False)
    return df

def to_total_pence(value):
    '''
    Unit,Composition,Total Pence (d)
    1 Pound (£),20 Shillings,240d
    1 Shilling (s),12 Pence,12d
    1 Penny (d),4 Farthings,1d
    '''
    value = value.split(' ')
    pounds = int(value[0])
    shillings = int(value[1])
    pence = int(value[2])
    return pounds * 240 + shillings * 12 + pence


def from_total_pence(total_pence):
    pounds = total_pence // 240
    remaining_pence = total_pence % 240
    
    shillings = remaining_pence // 12
    pence = remaining_pence % 12
    
    return pounds, shillings, pence

def check_if_correct(total_land_val, total_total_val, sum_land_val, sum_total_val):
    
    def off_by_factor_of_two(a, b):
        return a == 2 * b or b == 2 * a
    
    return off_by_factor_of_two(total_land_val, sum_land_val) and off_by_factor_of_two(total_total_val, sum_total_val)


def get_total_values(df, townland):
    '''
    df = df[townland] -> get the df for the townland
    total_land_val, total_total_val = value of land and total for the townland at the row where is_total = 1
    '''
    df_townland = df[df['townland'] == townland]
    total_land_val = to_total_pence(df_townland[df_townland['is_total'] == 1]['land_val'].values[0])
    total_total_val = to_total_pence(df_townland[df_townland['is_total'] == 1]['total_val'].values[0])

    sum_land_val, sum_total_val = 0 , 0 
    for index, row in df_townland[df_townland['is_total'] == 0].iterrows():
        sum_land_val += to_total_pence(row['land_val'])
        sum_total_val += to_total_pence(row['total_val'])
    is_correct = check_if_correct(total_land_val, total_total_val, sum_land_val, sum_total_val)
    sum_land_val, sum_total_val = from_total_pence(sum_land_val), from_total_pence(sum_total_val)

    return {'sum_land_val': sum_land_val, 'total_land_val': total_land_val, 'sum_total_val': sum_total_val, 'total_total_val': total_total_val, 'is_correct': is_correct}
