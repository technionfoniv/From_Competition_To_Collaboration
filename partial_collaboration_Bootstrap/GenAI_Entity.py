import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd
from sklearn.naive_bayes import ComplementNB
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score
import numpy as np
from scipy.stats import loguniform, randint
import numpy as np
import re
os.environ["TOKENIZERS_PARALLELISM"] = "true"



def find_threshold_precision_twice_recall(y_true, y_probs, ratio=2.0):
    """
    Finds the threshold where Precision >= ratio * Recall,
    and among those, picks the threshold with the highest Recall.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_probs : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    ratio : float, default=2.0
        Required Precision / Recall ratio.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # thresholds aligns with recalls[1:], precisions[1:]
    precisions, recalls, thresholds = precisions[1:], recalls[1:], thresholds

    # mask for the constraint: P >= ratio * R
    mask = precisions >= ratio * recalls

    if not np.any(mask):
        print(f"No threshold found where Precision >= {ratio} × Recall.")
        return 0.5  # fallback default

    # among feasible points, maximize recall
    best_idx = np.argmax(recalls[mask])
    chosen = np.where(mask)[0][best_idx]

    best_threshold = thresholds[chosen]
    best_precision = precisions[chosen]
    best_recall = recalls[chosen]

    print(f"Chosen Threshold={best_threshold:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f} (Precision ≈ {ratio}× Recall)")
    return best_threshold
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class genAI_Entity:
    def __init__(self, model_name="../genAI_classifier", num_labels=2):
    
        self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
        # self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
        self.dataset=None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words='english'
        )
        self.best_model = None
    def load_model(self, base_dir, llm, subject, subsample):
        """
        Loads the model with the highest week_index for the given parameters.
        Returns None if no model is found.
        """
        # Construct the search directory up to the subsample level
        search_dir = os.path.join(
            base_dir, 
            str(llm), 
            str(subject), 
            f"subsample_{subsample}"
        )

        # Check if the directory exists
        if not os.path.exists(search_dir):
            print(f"No directory found at: {search_dir}")
            return None

        # Find all subdirectories that look like 'week_X'
        max_week = -1
        target_week_dir = None

        try:
            for item in os.listdir(search_dir):
                full_path = os.path.join(search_dir, item)
                
                # Check if it is a directory and matches the 'week_' pattern
                if os.path.isdir(full_path) and item.startswith("week_"):
                    # Extract the integer week number using regex or split
                    try:
                        week_num = int(item.split('_')[1])
                        if week_num > max_week:
                            max_week = week_num
                            target_week_dir = item
                    except (IndexError, ValueError):
                        continue # Skip folders that don't match the format exactly
            
            if target_week_dir:
                load_path = os.path.join(search_dir, target_week_dir, "model.pkl")
                if os.path.exists(load_path):
                    with open(load_path, 'rb') as f:
                        instance = pickle.load(f)
                    print(f"Loaded model from week {max_week}: {load_path}")
                    return instance
                else:
                    print(f"Directory found for week {max_week}, but 'model.pkl' is missing.")
                    return None
            else:
                print("No week directories found.")
                return None

        except Exception as e:
            print(f"Error loading model: {e}")
            return None    
    def save_model(self, base_dir, llm, subject, week_index, subsample):
        """
        Saves the current instance of the class to a specific path based on parameters.
        Structure: base_dir/llm/subject/subsample_{subsample}/week_{week_index}/model.pkl
        """
        # Construct the directory path
        # We cast variables to string to ensure path concatenation works
        save_path = os.path.join(
            base_dir, 
            str(llm), 
            str(subject), 
            f"subsample_{subsample}", 
            f"week_{week_index}"
        )

        # Create the directories if they don't exist
        os.makedirs(save_path, exist_ok=True)

        # Define the full file path
        file_path = os.path.join(save_path, "model.pkl")

        # Save the object using pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Model successfully saved to: {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    
    def create_dataset(self,sub, llm, max_week_index, target_subsample):
        """
        Reads parquet files where the file's week_index is <= max_week_index
        and the file's subsample is equal to target_subsample.
        """
        # 1. Define the directory path based on arguments
        # Note: Refactored to list files from the SAME directory used for reading
        directory_path = f'bootstrap_results_G-Utility'
        
        dfs = []
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return pd.DataFrame()


        for filename in os.listdir(directory_path):
                if not filename.endswith(".parquet"):
                    continue
                if (sub not in filename) or (llm not in filename):
                    continue
                # 3. Parse Metadata using Regex
                # Pattern matches: ...sample_1_week_1.parquet
                # We capture the digits (\d+) after 'sample_' and 'week_'
                sample_match = re.search(r'sample_(\d+)', filename)
                week_match = re.search(r'week_(\d+)', filename)

                if sample_match and week_match:
                    try:
                        file_subsample = int(sample_match.group(1))
                        file_week = int(week_match.group(1))

                        # 4. Filter Logic
                        # Check if this file belongs to the correct subsample AND is within the week range
                        if (file_subsample == target_subsample) and (file_week <= max_week_index):
                            
                            file_path = os.path.join(directory_path, filename)
                            df = pd.read_parquet(file_path)
                            dfs.append(df)
                            
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

            # 5. Concatenate Results
        if dfs:
            dataset = pd.concat(dfs, ignore_index=True)
            self.dataset = TextDataset(
                dataset['text'].tolist(),
                dataset['genai_label'].astype(int).tolist(),
                self.tokenizer
            )
        else:
            self.dataset = pd.DataFrame()
                
    
    
    def train_with_cv(self,  optimize_for="recall", n_iter=20):
        texts=self.dataset.texts
        labels=np.array(self.dataset.labels)
        scoring = make_scorer(recall_score) if optimize_for == "recall" else "roc_auc"

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")),
            ("nb", ComplementNB())  # Optimized for imbalanced text data
        ])

        param_distributions = {
            "nb__alpha": loguniform(1e-3, 1e1)  # Smoothing factor
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        search.fit(texts, labels)
        print("✅ Best Params (Recall Optimized):", search.best_params_)
        print("✅ Best CV Recall:", search.best_score_)

        self.best_model = search.best_estimator_ 
         
    def predict_proba(self, texts):
        return self.best_model.predict_proba(texts)[:, 1]   
       