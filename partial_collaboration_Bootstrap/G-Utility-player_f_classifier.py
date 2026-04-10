import pandas as pd
from tqdm import tqdm
from GenAI_Entity import genAI_Entity
from Forums_Entity import Forum_Entity
import os
import warnings
from Thresholds import Thresholds
import re
import logging

# ---------------------------------------------------------
# 1. Setup Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_g_utility_player_f_classifier_execution.log", mode='a'), # Dedicated log file
        logging.StreamHandler()                                            # Prints to console
    ]
)
logger = logging.getLogger(__name__)
#Utilites
GEN_AI_UTILITY='perplexity'
Forum_UTILITY='NormalizedViewCount'
Forum_UTILITY_NORMALIZED='ViewCount'
warnings.filterwarnings("ignore")


def get_total_tagged_data_so_far(sub, llm, max_week_index, target_subsample):
    """
    Reads parquet files where the file's week_index is <= max_week_index
    and the file's subsample is equal to target_subsample.
    """
    directory_path = f'bootstrap_results_G-Utility'
    
    dfs = []
    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        return pd.DataFrame()


    for filename in os.listdir(directory_path):
            if not filename.endswith(".parquet"):
                continue
            if (sub not in filename) or (llm not in filename):
                continue

            sample_match = re.search(r'sample_(\d+)', filename)
            week_match = re.search(r'week_(\d+)', filename)

            if sample_match and week_match:
                try:
                    file_subsample = int(sample_match.group(1))
                    file_week = int(week_match.group(1))

                    if (file_subsample == target_subsample) and (file_week <= max_week_index):
                        
                        file_path = os.path.join(directory_path, filename)
                        df = pd.read_parquet(file_path)
                        dfs.append(df)
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")

    if not dfs:
            return pd.DataFrame()
    total_data = pd.concat(dfs, ignore_index=True)
    return total_data

 

def greedy_genAI_turn(weekly_data):
    weekly_data['Model_Score']=weekly_data[GEN_AI_UTILITY]
    weekly_data=weekly_data.sort_values(by='Model_Score', ascending=False).head(MAXIMUM_NUMBER_OF_QUESTIONS_PER_WEEK_BY_GENAI)
    return weekly_data

def genAI_turn(weekly_data):
    weekly_data['model_proba'] = genAI.predict_proba(weekly_data['text'].to_list())
    # weekly_data['model_proba'] = weekly_data['model_proba'].apply(lambda x: x[1])
    weekly_data['Model_Score'] = weekly_data[GEN_AI_UTILITY] * weekly_data['model_proba']
    weekly_data = weekly_data.sort_values(by='Model_Score', ascending=False).head(MAXIMUM_NUMBER_OF_QUESTIONS_PER_WEEK_BY_GENAI)
    return weekly_data


def forum_turn(weekly_data):
    weekly_data['pred_class'] = forum.predict_batch(weekly_data['text'].to_list())
    weekly_data['forum_threshold'] = forum.threshold
    weekly_data['forum_proba'] = forum.predict_proba_batch(weekly_data['text'].to_list())
    weekly_data['forum_proba'] = weekly_data['forum_proba'].apply(lambda x: x[1])

    weekly_data = weekly_data.sort_values(by='forum_proba', ascending=False)

    # --- 2. forum_label (New addition) ---
    weekly_data['forum_label'] = False
    weekly_data.iloc[:MAXIMUM_NUMBER_OF_QUESTIONS_PER_WEEK_BY_FORUM, 
                     weekly_data.columns.get_loc('forum_label')] = True
    weekly_data['forum_label'] = weekly_data['forum_label'] & (weekly_data['forum_proba'] > weekly_data['forum_threshold'])
    weekly_data['genai_label'] = weekly_data['forum_label'].astype(int) & weekly_data['AnswerCount'].gt(0)

    return weekly_data

def save_weekly_results(weekly_data, week_index,subsample, sub, llm):
        output_directory = f'bootstrap_results_G-Utility'
        os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(output_directory, f'weekly_data_week_{week_index}_sample_{subsample}_subject_{sub}_{llm}.parquet')
        weekly_data.to_parquet(file_path, index=False)
        return file_path


START_DATE = pd.to_datetime('2024-07-23')
MAXIMUM_NUMBER_OF_QUESTIONS_PER_WEEK_BY_FORUM = 50
MAXIMUM_NUMBER_OF_QUESTIONS_PER_WEEK_BY_GENAI = 100
SUBJECTS = ['united']
LLMS = [
"EleutherAI-pythia-6.9b",
"meta-llama-Llama-3.1-8B",
"meta-llama-Meta-Llama-3-8B-Instruct"
]
weeks=range(53)

logger.info("=== Starting G-Utility-Player-F-Classifier Pipeline Execution ===")

for llm in LLMS:       
    for sub in SUBJECTS:
        logger.info(f"Initializing models for Subject: {sub}, LLM: {llm}")
        genAI = genAI_Entity(model_name=f"../results/models/stackexchange_models_united_initial_genAI_partial_collaboration", num_labels=2)
        for week_index, week in tqdm(enumerate(weeks), desc=f"Processing weeks for {llm}"):
            if ((week_index+1) % 13 == 0 and week_index != 51) or week_index==0: 
                path=f"Player-F-Models/united_model_{week_index//13}"#fILE PATH FORUM MODEL
                forum = Forum_Entity(model_name=path, num_labels=2)
                forum.threshold = Thresholds.get(f"united_model_{week_index//13}")
                logger.info(f"Loaded Forum_Entity with model: {path} and threshold: {forum.threshold}")
            for i in range(50):
                # Context string for cleaner logs
                ctx = f"[{llm} | {sub} | Wk {week_index+1} | Spl {i+1}]"

                try:
                    genAI_data = pd.read_parquet(f'bootstrap_dataset/bootstrap_dataset_{sub}_{llm}_sample_{i+1}_week_{week_index+1}.parquet')
                except FileNotFoundError as e:
                    logger.error(f"{ctx} Dataset file not found: {e}")
                    continue

                genAI_data['AnswerCount'] = genAI_data['AnswerBody'].apply(lambda x: 1 if isinstance(x, str) and x.strip() != '' else 0)
                weekly_data = genAI_data[['text', GEN_AI_UTILITY,Forum_UTILITY,'CreationDate','AnswerCount','QuestionId','year_week']].copy().drop_duplicates().reset_index(drop=True)
                sum_view_count = weekly_data[Forum_UTILITY].sum()
                weekly_data[Forum_UTILITY_NORMALIZED] = weekly_data[Forum_UTILITY] / sum_view_count
                logger.info(f"{ctx} Total questions loaded: {len(weekly_data)}")
                
                if week_index<13:
                    weekly_data = greedy_genAI_turn(weekly_data)
                else:
                    logger.info(f"{ctx} Loading advanced model for GenAI turn")
                    genAI.best_model=genAI.load_model('Player-G-Models', llm, sub, i+1)    
                    weekly_data = genAI_turn(weekly_data)
                    
                weekly_data = forum_turn(weekly_data)
                
                genai_labeled_count = len(weekly_data[weekly_data['genai_label']])
                total_perp_week = weekly_data[weekly_data['genai_label']][GEN_AI_UTILITY].sum()
                
                logger.info(f"{ctx} GenAI labeled questions this week: {genai_labeled_count}")
                logger.info(f"{ctx} Total {GEN_AI_UTILITY} achieved this week: {total_perp_week:.4f}")
                
                save_weekly_results(weekly_data, week_index+1,i+1, sub, llm)
                
                total_results_df= get_total_tagged_data_so_far(sub,llm,week_index+1,i+1)
                
                if not total_results_df.empty:
                    total_results=total_results_df[(total_results_df['genai_label']==True)&(total_results_df['AnswerCount'].gt(0))][GEN_AI_UTILITY].sum()
                    true_labels_count = len(total_results_df[total_results_df['genai_label']==True])
                    true_labels_with_answers = len(total_results_df[(total_results_df['AnswerCount'].gt(0))&(total_results_df['genai_label']==True)])
                    
                    logger.info(f"{ctx} Total {GEN_AI_UTILITY} across all weeks: {total_results:.4f}")
                    logger.info(f"{ctx} Cumulative GenAI labels: {true_labels_count} | With answers: {true_labels_with_answers}")
                else:
                    logger.warning(f"{ctx} total_results_df is empty. Cannot calculate cumulative metrics.")

                # Model Training Logic Logging
                if week_index !=0 and (week_index+1) % 13 == 0 and week_index<52:
                    logger.info(f"{ctx} Triggering model training cycle")
                    genAI.create_dataset(sub, llm, week_index+1, i+1)
                    genAI.train_with_cv() 
                    genAI.save_model('Player-G-Models', llm, sub, week_index+1,i+1)
                    logger.info(f"{ctx} Model training and save complete")

            # End of sample loop (50 samples per week)
            total_results_df = get_total_tagged_data_so_far(sub,llm,week_index+1, i+1) # Using last sample index for summary
            if not total_results_df.empty:
                total_results = total_results_df[(total_results_df['genai_label'] == True) & (total_results_df['AnswerCount'].gt(0))][GEN_AI_UTILITY].sum()
                logger.info(f"--- WEEK {week_index+1} SUMMARY [{llm}] ---")
                logger.info(f"Total cumulative {GEN_AI_UTILITY}: {total_results:.4f}")
                logger.info(f"Cumulative GenAI labels: {len(total_results_df[total_results_df['genai_label'] == True])}")
                logger.info(f"Cumulative GenAI labels with answers: {len(total_results_df[(total_results_df['AnswerCount'].gt(0)) & (total_results_df['genai_label'] == True)])}")

logger.info("=== G-Utility-Player-F-Classifier Pipeline Execution Completed ===")