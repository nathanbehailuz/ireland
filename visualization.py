import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Helper function to convert "£ s d" strings back to pence for numerical plotting
def parse_currency_to_pence(value_str):
    if pd.isna(value_str) or value_str == 0:
        return 0
    try:
        parts = str(value_str).split()
        if len(parts) != 3:
            return 0
        pounds = int(parts[0])
        shillings = int(parts[1])
        pence = int(parts[2])
        return (pounds * 240) + (shillings * 12) + pence
    except:
        return 0

# Load data
try:
    df = pd.read_csv('./comparison_results.csv')
    # Convert string "True"/"False" to boolean/numeric (1/0)
    # First try reading as boolean directly, fall back to string mapping if needed
    if df['is_correct'].dtype == bool:
        df['is_correct'] = df['is_correct'].astype(int)
    else:
        # Handle string values and convert to 1/0, treating any NaN or unexpected values as 0
        df['is_correct'] = df['is_correct'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
except FileNotFoundError:
    print("Error: results/comparison_results.csv not found. Please run compare_llms.py first.")
    exit()

# Set visual style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Overall Accuracy Bar Chart
# ==========================================
plt.figure(figsize=(8, 6))
accuracy_df = df.groupby('llm')['is_correct'].mean().reset_index()
accuracy_df['accuracy_pct'] = accuracy_df['is_correct'] * 100

ax = sns.barplot(data=accuracy_df, x='llm', y='accuracy_pct', palette='viridis')
plt.title('Overall Extraction Accuracy by LLM', fontsize=16)
plt.ylabel('Accuracy (% Correct Sums)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 100)

# Add percentage labels on bars
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f%%', padding=3)

plt.savefig('results/viz_accuracy_bar.png')
print("Saved Overall Accuracy Chart")

# ==========================================
# 2. Correctness Heatmap
# ==========================================
plt.figure(figsize=(10, len(df['townland'].unique()) * 0.4 + 2))  # Adjust height based on data size

# Pivot data for heatmap
heatmap_data = df.pivot_table(
    index=['page', 'townland'], 
    columns='llm', 
    values='is_correct',
    aggfunc='first' # Assuming one entry per townland per llm
)

# Create heatmap (1 = Green/Correct, 0 = Red/Incorrect)
cmap = sns.color_palette(["#e74c3c", "#2ecc71"]) # Red to Green
sns.heatmap(heatmap_data, annot=True, cmap=cmap, cbar=False, linewidths=.5, fmt='.0f')

plt.title('Correctness Heatmap per Townland', fontsize=16)
plt.tight_layout()
plt.savefig('results/viz_heatmap.png')
print("Saved Heatmap")

# ==========================================
# 3. Error Magnitude Scatter Plot
# ==========================================
# Convert currency strings to integers (pence) for plotting
df['val_calculated_pence'] = df['sum_total_val'].apply(parse_currency_to_pence)
df['val_extracted_pence'] = df['total_total_val'].apply(parse_currency_to_pence)

# Add small jitter to separate overlapping points
np.random.seed(42)  # For reproducibility
jitter_strength = 0.02  # 2% jitter
df['x_jittered'] = df['val_extracted_pence'] * (1 + np.random.uniform(-jitter_strength, jitter_strength, len(df)))
df['y_jittered'] = df['val_calculated_pence'] * (1 + np.random.uniform(-jitter_strength, jitter_strength, len(df)))

plt.figure(figsize=(10, 8))

# Plot with jittered coordinates
sns.scatterplot(
    data=df, 
    x='x_jittered', 
    y='y_jittered', 
    hue='llm', 
    style='is_correct',
    s=150,  # Larger markers
    alpha=0.8,  # More opaque
    edgecolor='black',  # Add edge for better visibility
    linewidth=0.5
)

# Add a diagonal line (Perfect match line)
max_val = max(df['val_extracted_pence'].max(), df['val_calculated_pence'].max())
plt.plot([0, max_val], [0, max_val], ls="--", c=".3", label="Perfect Match")

# Count total points
total_points = len(df)
plt.title(f'Calculated Sum vs. Extracted Total (Log Scale)', fontsize=16)
plt.xlabel('Extracted Total Value (Pence)', fontsize=12)
plt.ylabel('Calculated Sum of Rows (Pence)', fontsize=12)
plt.legend(title='Model / Correctness', loc='upper left')

# Use log scale if values vary wildly (likely in historical property data)
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.savefig('results/viz_scatter_deviation.png', dpi=150)
print(f"Saved Scatter Plot ({total_points} data points plotted)")

# Print breakdown for verification
print(f"\nData point breakdown:")
for llm in sorted(df['llm'].unique()):
    count = len(df[df['llm'] == llm])
    print(f"  {llm}: {count} points")