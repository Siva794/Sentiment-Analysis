# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "surprise": 5
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset
    """
    df = df.dropna().copy()

    if "text" not in df.columns:
        df.columns = ["text", "label"]

    df.loc[:, "label"] = df["label"].str.strip().str.lower()

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode labels safely
    """
    df.loc[:, "label"] = df["label"].map(LABEL_MAP)

    df = df.dropna(subset=["label"]).copy()
    df.loc[:, "label"] = df["label"].astype(int)

    return df


def reduce_dataset(df: pd.DataFrame, sample_size: int = 50000) -> pd.DataFrame:
    """
    Reduce dataset size
    """
    return df.sample(n=sample_size, random_state=42)


def split_data(df: pd.DataFrame, test_size: float = 0.1):
    """
    Split dataset
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    return train_df, test_df


def save_data(train_df, test_df, output_dir="data/processed"):
    """
    Save processed data
    """
    train_path = f"{output_dir}/train.csv"
    test_path = f"{output_dir}/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train saved: {train_path}")
    print(f"✅ Test saved: {test_path}")