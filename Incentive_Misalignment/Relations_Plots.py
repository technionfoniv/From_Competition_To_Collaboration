import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

# ==========================================
# CONSTANTS
# ==========================================

# Base configurations
BASE_DIR = "results"
SUBJECTS = {'ubuntu', 'english', 'latex', 'math', 'stackoverflow'} # Set used to ensure uniqueness
LLMS = [
    "EleutherAI-pythia-6.9b",
    "meta-llama-Llama-3.1-8B",
    "meta-llama-Meta-Llama-3-8B-Instruct"
]

# File paths & thresholds
FILE_PATH_TEMPLATE = f"{BASE_DIR}/stackexchange_{{subject}}_combined/{{llm}}/aligned.parquet"
CUTOFF_DATE = "2024-07-23"

# Column Names
COL_DATE = 'CreationDate'
COL_U_F = 'ViewCount'
COL_DATE_OLD = 'Question_Creation_Date'
COL_U_G = 'perplexity'
COL_Q_ID = 'QuestionId'
COL_YEAR = 'Year'
COL_WEEK = 'week'
COL_SCALED_VIEW = 'ScaledViewCount'

# Column groupings
REQUIRED_COLS_LOAD = [COL_U_G, COL_U_F, COL_DATE, COL_Q_ID]
REQUIRED_COLS_PLOT = [COL_U_G, COL_U_F, COL_YEAR, COL_WEEK]

# Plotting constants
PLOT_NCOLS = 2
PLOT_HEIGHT_PER_ROW = 5
PLOT_WIDTH = 18
PLOT_SCATTER_SIZE = 20
PLOT_SCATTER_ALPHA = 0.6
PLOT_TITLE_FONTSIZE = 16
PLOT_ANNOTATION_FONTSIZE = 10

# ==========================================
# DATA LOADING
# ==========================================

# Initialize the output dictionary
llm_data = {llm: {} for llm in LLMS}

for subject in SUBJECTS:
    for llm in LLMS:
        file_path = FILE_PATH_TEMPLATE.format(subject=subject, llm=llm)
        
        # Best practice: Check if file exists to prevent crashes
        if not os.path.exists(file_path):
            print(f"Warning: File not found -> {file_path}")
            continue

        df = pd.read_parquet(file_path)

        # Standardize column names
        rename_mapping = {}
        if COL_DATE not in df.columns and COL_DATE_OLD in df.columns:
            rename_mapping[COL_DATE_OLD] = COL_DATE

            
        if rename_mapping:
            df = df.rename(columns=rename_mapping)

        # Process dates
        if COL_DATE in df.columns:
            df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')


        # Keep only required columns and drop duplicates
        available_cols = [c for c in REQUIRED_COLS_LOAD if c in df.columns]
        df = df[available_cols].drop_duplicates()
        
        llm_data[llm][subject] = df

# ==========================================
# PLOTTING
# ==========================================

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

for llm, subject_dfs in llm_data.items():
    if not subject_dfs:
        continue

    n_subjects = len(subject_dfs)
    nrows = (n_subjects + PLOT_NCOLS - 1) // PLOT_NCOLS

    fig, axes = plt.subplots(nrows=nrows, ncols=PLOT_NCOLS, figsize=(PLOT_WIDTH, nrows * PLOT_HEIGHT_PER_ROW))
    fig.suptitle(f"Scatter plots of Perplexity vs Week-wise Scaled ViewCount for {llm}", fontsize=PLOT_TITLE_FONTSIZE)
    
    # Flatten axes array for easy iteration, handling cases where it's not a 2D array
    axes = axes.flatten() if isinstance(axes, plt.Axes) == False else [axes]

    for i, (subject, df) in enumerate(subject_dfs.items()):
        ax = axes[i]
        df = df.copy()

        # Extract year & week if date column exists
        if COL_DATE in df.columns:
            df[COL_YEAR] = df[COL_DATE].dt.year
            df[COL_WEEK] = df[COL_DATE].dt.isocalendar().week

        # Check if we have the necessary columns for plotting
        if all(col in df.columns for col in REQUIRED_COLS_PLOT):
            df = df[REQUIRED_COLS_PLOT].dropna()

            # Apply min-max scaling PER (Year, Week)
            df[COL_SCALED_VIEW] = df.groupby([COL_YEAR, COL_WEEK])[COL_U_F].transform(
                lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
            )

            if not df.empty:
                sns.scatterplot(
                    data=df, x=COL_U_G, y=COL_SCALED_VIEW, 
                    ax=ax, s=PLOT_SCATTER_SIZE, alpha=PLOT_SCATTER_ALPHA
                )
                ax.set_title(subject)
                ax.set_ylabel("Scaled ViewCount (by week)")

                # Spearman correlation
                corr, _ = spearmanr(df[COL_U_G], df[COL_SCALED_VIEW])
                ax.annotate(f"Spearman: {corr:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=PLOT_ANNOTATION_FONTSIZE)
            else:
                ax.set_title(subject)
                ax.annotate("No valid data for these weeks", xy=(0.05, 0.5), xycoords='axes fraction', fontsize=PLOT_ANNOTATION_FONTSIZE)
        else:
            ax.set_title(f"{subject} (missing cols)")
            ax.annotate("Missing required columns", xy=(0.05, 0.5), xycoords='axes fraction', fontsize=PLOT_ANNOTATION_FONTSIZE)

    # Clean up empty subplots
    for j in range(len(subject_dfs), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()