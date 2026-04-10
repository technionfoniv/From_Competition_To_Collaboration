import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import logging
from Forums_Entity import Forum_Entity

# ---------------------------------------------------------
# 1. Setup Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("heuristic_selection_clf.log", mode='a'), # Dedicated log file
        logging.StreamHandler()                                   # Prints to console
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
#   GLOBAL SCORE VARIABLE
# ============================================================
Player_F_UTILITY_NORM = "min_max_normalized_viewCount"
Player_F_Utility="ViewCount"
# ============================================================

# TODO: Load your classifier here. 
# Example:
# import joblib
# clf = joblib.load('my_classifier.pkl')

def fair_round_robin(df, k, clf=None):
    df = df.copy()
    # "proba" is now already predicted and attached to df in the main loop

    # Normalize UTILITY for fair comparison 
    perp = df[Player_G_UTILITY].values
    perp_util = (perp - perp.min()) / (perp.max() - perp.min())
    df[Player_G_UTILITY + "_util"] = perp_util

    # Build two independent rankings
    perp_rank = df.sort_values(Player_G_UTILITY + "_util", ascending=False).index.tolist()
    proba_rank = df.sort_values("proba", ascending=False).index.tolist()

    selected = []
    i_perp, i_proba = 0, 0

    # Alternate picking until k unique items are selected
    while len(selected) < k and (i_perp < len(perp_rank) or i_proba < len(proba_rank)):
        if len(selected) % 2 == 0:  # even → take from UTILITY
            while i_perp < len(perp_rank) and perp_rank[i_perp] in selected:
                i_perp += 1
            if i_perp < len(perp_rank):
                selected.append(perp_rank[i_perp])
                i_perp += 1
        else:  # odd → take from probability
            while i_proba < len(proba_rank) and proba_rank[i_proba] in selected:
                i_proba += 1
            if i_proba < len(proba_rank):
                selected.append(proba_rank[i_proba])
                i_proba += 1

    return selected



def max_product(df, k, clf=None):
    perp = df[Player_G_UTILITY].values
    perp_util = (perp - perp.min()) / (perp.max() - perp.min())
    df = df.copy()
    df[Player_G_UTILITY + "_util"] = perp_util

    # "proba" is now the classifier probability instead of Player_F_UTILITY_NORM
    df["product"] = df[Player_G_UTILITY + "_util"] * df["proba"]

    return df.sort_values("product", ascending=False).head(k).index.tolist()



def random_choice(df, k, clf=None):
    return df.sample(min(k, len(df)), random_state=42).index.tolist()


def greedy_nash_bargaining(df: pd.DataFrame, k: int):
    # Convert columns to numpy arrays for raw speed
    perp = df[Player_G_UTILITY].values
    # Avoid division by zero if max == min
    p_min, p_max = perp.min(), perp.max()
    perp_util = (perp - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(perp)
    
    # Replaced Player_F_Utility with the classifier 'proba'
    proba = df["proba"].values
    indices = df.index.values
    
    # Track selection state
    n = len(df)
    k = min(k, n)
    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    
    current_u_a = 0.0
    current_u_b = 0.0
    
    for _ in range(k):
        # Calculate Nash product for all rows at once, mask out already selected
        # (current_u_a + potential_a) * (current_u_b + potential_b)
        nash_products = (current_u_a + perp_util) * (current_u_b + proba)
        
        # Set already selected items to -infinity so they aren't picked again
        nash_products[selected_mask] = -np.inf
        
        # Find the best remaining candidate
        best_local_idx = np.argmax(nash_products)
        
        # Update running totals and selection state
        current_u_a += perp_util[best_local_idx]
        current_u_b += proba[best_local_idx]
        selected_mask[best_local_idx] = True
        selected_indices.append(indices[best_local_idx])
        
    return selected_indices

def normalize_group(group):
    scaler = MinMaxScaler()
    group[Player_F_UTILITY_NORM] = scaler.fit_transform(group[[Player_F_Utility]])
    return group


SUBJECTS = ['united']
LLMS = [
    "EleutherAI-pythia-6.9b",
    "meta-llama-Llama-3.1-8B",
    "meta-llama-Meta-Llama-3-8B-Instruct"
]

k = 50
weeks=range(53)
PATH_TO_MODEL=f"../partial_collaboration_Bootstrap/Player-F-Models/united_model_{{week_index}}"
Player_G_UTILITY='perplexity'

logger.info("=== Starting Heuristic Selection Pipeline ===")
logger.info(f"Total weeks to process: {len(weeks)}")

for llm in LLMS:
    for week_index, week in tqdm(enumerate(weeks), desc=f"Processing weeks for {llm}"):
        if ((week_index+1) % 13 == 0 and week_index != 51) or week_index==0: 
            path=PATH_TO_MODEL.format(week_index=week_index//13)
            clf = Forum_Entity(model_name=path, num_labels=2)

        for sample in range(50):

            ctx = f"[{llm} | Spl {sample+1} | Wk {week_index+1}"
            
            try:
                df = pd.read_parquet(f'bootstrap_dataset/bootstrap_dataset_united_{llm}_sample_{sample+1}_week_{week_index+1}.parquet')
            except FileNotFoundError as e:
                logger.error(f"{ctx} Dataset file not found: {e}")
                continue
            
            # Note: AnswerBody might be missing if the schema is inconsistent, wrapped in a quick try/except just in case
            try:
                df['AnswerCount'] = df['AnswerBody'].apply(lambda x: 1 if isinstance(x, str) and x.strip() != '' else 0)
            except KeyError:
                logger.warning(f"{ctx} 'AnswerBody' column missing. 'AnswerCount' not generated.")
            
            try:
                df = df[["text", Player_G_UTILITY, Player_F_Utility, "CreationDate", "QuestionId",'AnswerCount']].drop_duplicates().reset_index(drop=True)
                sum_UTILITIES = df[Player_F_Utility].sum()
                df[Player_F_Utility + "_normalized"] = df[Player_F_Utility] / sum_UTILITIES
            except KeyError as e:
                logger.error(f"{ctx} Missing required columns for filtering: {e}")
                continue
                
            df["CreationDate"] = pd.to_datetime(df["CreationDate"])
            df = normalize_group(df) 
            # --- PREDICT PROBABILITY ---
            # Using the classifier to predict the probability for the "text" column
            # (Assuming clf is defined above and predict_proba returns a 2D array where index 1 is the positive class)
            df['proba'] = clf.predict_proba_batch(df['text'].to_list())
            df['proba'] = df['proba'].apply(lambda x: x[1])


            
            methods = {
                "fair_round_robin":  fair_round_robin,
                "max_product": max_product,
                "random_choice": random_choice,
                "nash_bargaining": greedy_nash_bargaining,
            }
            
            os.makedirs('selected_questions', exist_ok=True)
            
            for method_name, func in methods.items():
                selected_ids = func(df, k)
                selected = df.loc[selected_ids].copy()
                selected["method"] = method_name
                selected["week"] = week
                selected["llm"] = llm
                selected["sample"] = sample

                output_path = f'selected_questions/selected_{llm}_sample_{sample+1}_week_{week_index+1}_{method_name}.parquet'
                
                selected.to_parquet(output_path, index=False)
                logger.info(f"{ctx} Saved {len(selected)} questions via {method_name} to {output_path}")
                # logger.info(f"{ctx} {method_name} - Selected Question IDs: {selected['QuestionId'].tolist()}")
                
                # Updated logging to reflect the predicted probability as well
                logger.info(f"{ctx} {method_name}- total {Player_G_UTILITY}: {selected[Player_G_UTILITY].sum():.2f} - total proba: {selected['proba'].sum():.2f} - total PLAYER F UTILITY: {selected[Player_F_Utility + '_normalized'].sum():.4f}")

logger.info("=== Heuristic Selection Pipeline Completed ===")