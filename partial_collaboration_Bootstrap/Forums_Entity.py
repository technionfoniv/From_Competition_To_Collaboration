import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score ,precision_recall_curve
import pandas as pd
import numpy as np
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

class Forum_Entity:
    def __init__(self, model_name="../sof_classifier",threshold=0.83, num_labels=2 ):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
        self.dataset=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.threshold = threshold
        self.metric="normalized_viewCount"
    def predict_batch(self, questions, batch_size=32):
        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            ).to(self.device)
            self.model.eval()

            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                results.extend((probs > self.threshold).cpu().tolist())
        return results
    def predict_proba_batch(self, questions, batch_size=32):
        results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            ).to(device)

            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                results.extend(probs.cpu().numpy())
        return results
    
    def read_initial_data(self,sub,llm):
        DATA_START = pd.to_datetime("2024-04-23")
        DATA_END=DATA_START + pd.DateOffset(months=3)
        initial_df = pd.read_parquet(f'../results/stackexchange_{sub}_combined/{llm}/divided/forum.parquet')
        if 'CreationDate' not in initial_df.columns or 'ViewCount' not in initial_df.columns :
                    initial_df = initial_df.rename(columns={'Question_Creation_Date':'CreationDate','QuestionViewCount':'ViewCount'})
        initial_df["CreationDate"] = pd.to_datetime(initial_df["CreationDate"])
        initial_df = initial_df[(initial_df['CreationDate'] >= DATA_START) & (initial_df['CreationDate'] < DATA_END)]
        initial_df['year_week'] = initial_df['CreationDate'].dt.to_period('W')
        initial_df['AnswerCount']=initial_df['AnswerBody'].apply(lambda x: 1 if isinstance(x, str) and x.strip() != '' else 0)
        initial_df=initial_df[['text', 'perplexity','ViewCount','CreationDate','AnswerCount','QuestionId','year_week']].copy().drop_duplicates().reset_index(drop=True)
        return initial_df
        
    def create_dataset(self,path,sub,llm):
        dfs = [self.read_initial_data(sub,llm)]
        os.makedirs(path, exist_ok=True)
        for filename in os.listdir(path):
            if filename.endswith(".parquet"):
                file_path = os.path.join(path, filename)
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if dfs:
            dataset = pd.concat(dfs, ignore_index=True)
            dataset['year'] = dataset['CreationDate'].dt.year
            dataset['week'] = dataset['CreationDate'].dt.isocalendar().week

            # Normalize view counts by year+week
            dataset[self.metric] = dataset.groupby(['year', 'week'])['ViewCount'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

            # Define threshold (example: 1 std above weekly mean)
            FORUM_THRESHOLD =  dataset[self.metric].median()

            lower_bound = dataset[self.metric].quantile(0.4)
            upper_bound = dataset[self.metric].quantile(0.6)

            dataset = dataset[
                (dataset[self.metric] <= lower_bound) |
                (dataset[self.metric] >= upper_bound)
            ].copy()


            # Relabel
            # Label
            dataset["forum_label"] = dataset[self.metric] > FORUM_THRESHOLD

            self.dataset = TextDataset(
                dataset['text'].tolist(),
                dataset['forum_label'].astype(int).tolist(),
                self.tokenizer
            )
        else:
            self.dataset = pd.DataFrame()
# Set to an empty DataFrame if no files are found
    def load_model(self,path):
        self.model=AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.threshold=0.811      
    def train_bert_binary(self,week, epochs=3, batch_size=100, lr=1e-5,path='forum_checkpoints'):
        print("Training forums model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Split train into train+val
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        X_train=self.dataset.texts
        y_train=self.dataset.labels
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        train_dataset = TextDataset(X_tr, y_tr, self.tokenizer)
        val_dataset = TextDataset(X_val, y_val, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # Scheduler: linear warmup + decay
        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / max(1, num_warmup_steps), 1.0)
        )

        # ---- Training loop ----
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # stability
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # ---- Validation ----
            self.model.eval()
            val_loss = 0
            all_labels, all_probs = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()

                    probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            roc_auc = roc_auc_score(all_labels, all_probs)

            print(f"Epoch {epoch+1}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, "
                f"Val ROC-AUC={roc_auc:.4f}")

            # Save model checkpoint
            self.model.save_pretrained(f"{path}/forum_classifier_{week}")
            self.tokenizer.save_pretrained(f"{path}/forum_tokenizer_{week}")
            self.threshold = find_threshold_precision_twice_recall(all_labels, all_probs)     