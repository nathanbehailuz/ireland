import pandas as pd

def load_data():
    """Load the two data sources"""
    df_ground_truth = pd.read_excel('./nathan_to_fix.xlsx')
    df_results = pd.read_csv('./comparison_results.csv')
    return df_ground_truth, df_results

def calculate_true_c_townlands(df_ground_truth):
    """Calculate the true number of townlands to check per page"""
    # Filter for check_townland == 1 and count per page
    true_counts = df_ground_truth[df_ground_truth['check_townland'] == 1].groupby('target_filename').size().reset_index(name='true_c_townlands')
    true_counts = true_counts.rename(columns={'target_filename': 'page'})
    return true_counts

def calculate_llm_metrics(df_results):
    """Calculate identified and correct counts per LLM"""
    llms = ['claude', 'gemini', 'openai']
    
    # Calculate identified counts (total rows per page/llm)
    identified = df_results.groupby(['page', 'llm']).size().reset_index(name='identified')
    identified_pivot = identified.pivot(index='page', columns='llm', values='identified').reset_index()
    identified_pivot.columns = ['page'] + [f'identified_{llm}' for llm in identified_pivot.columns[1:]]
    
    # Calculate correct counts (where is_correct == True)
    correct = df_results[df_results['is_correct'] == True].groupby(['page', 'llm']).size().reset_index(name='correct')
    correct_pivot = correct.pivot(index='page', columns='llm', values='correct').reset_index()
    correct_pivot.columns = ['page'] + [f'correct_{llm}' for llm in correct_pivot.columns[1:]]
    
    # Merge identified and correct
    llm_metrics = identified_pivot.merge(correct_pivot, on='page', how='outer')
    
    return llm_metrics

def create_summary_table(df_ground_truth, df_results):
    """Create the full summary table"""
    # Get pages that appear in comparison_results.csv
    pages_to_analyze = df_results['page'].unique()
    
    # Get true counts only for pages in comparison_results
    true_counts = calculate_true_c_townlands(df_ground_truth)
    true_counts = true_counts[true_counts['page'].isin(pages_to_analyze)]
    
    # Get LLM metrics
    llm_metrics = calculate_llm_metrics(df_results)
    
    # Merge everything (left join to keep only pages from comparison_results)
    summary = true_counts.merge(llm_metrics, on='page', how='right')
    
    # Fill NaN with 0 and convert to int
    numeric_cols = [col for col in summary.columns if col != 'page']
    summary[numeric_cols] = summary[numeric_cols].fillna(0).astype(int)
    
    # Reorder columns for better readability
    column_order = ['page', 'true_c_townlands', 
                    'identified_claude', 'identified_gemini', 'identified_openai',
                    'correct_claude', 'correct_gemini', 'correct_openai']
    # Only include columns that exist
    column_order = [col for col in column_order if col in summary.columns]
    summary = summary[column_order]
    
    return summary

def add_total_row(df):
    """Add a TOTAL row at the bottom summing all numeric columns"""
    numeric_cols = [col for col in df.columns if col != 'page']
    totals = df[numeric_cols].sum()
    
    total_row = pd.DataFrame([['TOTAL'] + totals.tolist()], columns=df.columns)
    df_with_total = pd.concat([df, total_row], ignore_index=True)
    
    return df_with_total

def main():
    # Load data
    df_ground_truth, df_results = load_data()
    
    # Create summary table
    summary = create_summary_table(df_ground_truth, df_results)
    
    # Add total row
    summary = add_total_row(summary)
    
    # Display the table
    print("\n" + "="*100)
    print("LLM Performance Summary Table")
    print("="*100)
    print(summary.to_string(index=False))
    print("="*100)
    
    # Save to CSV
    summary.to_csv('./results/llm_summary_table.csv', index=False)
    print(f"\nTable saved to ./results/llm_summary_table.csv")
    
    return summary

if __name__ == "__main__":
    main()


