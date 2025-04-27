import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import load_data
from app.logger import logger  # ✅ Import logger for provenance

# Provenance metadata
MODEL_VERSION = "v1.0"
DATASET_VERSION = "final_processed_data.csv"
PIPELINE_VERSION = "offline_eval_1.0"


def load_final_model(model_filename="trained_model.pkl"):
    logger.info(
        f"Loading model: {model_filename} | Model Version: {MODEL_VERSION}"
    )  # Provenance log
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model


def compute_hit_rate(model, train_df, test_df, k=20, rating_threshold=4.0):
    # All unique items from both train and test (to simulate candidate items)
    all_items = pd.concat([train_df, test_df])["movie_id"].unique()

    # Group train and test data by user
    train_groups = train_df.groupby("user_id")
    test_groups = test_df.groupby("user_id")

    hit_users = 0
    total_users = 0

    # Iterate over each user in the test set
    for user_id, test_group in test_groups:
        # Define relevant items (those rated >= rating_threshold in the test set)
        relevant_items = set(
            test_group.loc[
                test_group["rating"] >= rating_threshold, "movie_id"
            ].unique()
        )

        # Get items the user already interacted with in the training set
        if user_id in train_groups.groups:
            train_items = set(train_groups.get_group(user_id)["movie_id"].tolist())
        else:
            train_items = set()

        # Predict for all candidate items not in the training set (simulate unseen items)
        predictions = []
        for item_id in all_items:
            if item_id not in train_items:
                pred = model.predict(user_id, item_id).est
                predictions.append((item_id, pred))

        if not predictions:
            continue

        # Sort predictions by estimated rating in descending order and select top k items
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [p[0] for p in predictions[:k]]

        total_users += 1
        # If at least one recommended item is relevant, count it as a hit
        if any(item in relevant_items for item in top_k_items):
            hit_users += 1

    hit_rate = hit_users / float(total_users) if total_users > 0 else 0.0
    return hit_rate


def main():
    logger.info(f"Provenance | Starting offline evaluation:")
    logger.info(
        f"Model Version: {MODEL_VERSION}, Dataset: {DATASET_VERSION}, Pipeline Version: {PIPELINE_VERSION}"
    )

    # Load the full dataset using your data.py
    df_all = load_data(f"data/{DATASET_VERSION}")
    logger.info(f"Loaded dataset with {len(df_all)} rows.")  # Provenance log

    # Sample a subset to keep the evaluation manageable; here we sample 2000 rows
    sample_size = 2000
    if len(df_all) > sample_size:
        df_all = df_all.sample(n=sample_size, random_state=42)
        logger.info(
            f"Sampled {sample_size} rows from the full dataset for offline evaluation."
        )  # Provenance log
    else:
        logger.info(
            "Dataset is smaller than or equal to 2000 rows; using the full dataset."
        )  # Provenance log

    # Split the sampled data into train and test sets (e.g., 80/20 split)
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    logger.info(
        f"Train set size: {len(train_df)} rows, Test set size: {len(test_df)} rows."
    )  # Provenance log

    # Load the final trained model (assumes the model was trained and saved as 'trained_model.pkl')
    svd_model = load_final_model("trained_models/trained_model.pkl")

    # Compute Hit Rate with top-20 recommendations using the specified rating threshold
    k = 20
    rating_threshold = 2.0
    hit_rate = compute_hit_rate(
        svd_model, train_df, test_df, k=k, rating_threshold=rating_threshold
    )

    logger.info(
        f"Evaluation Metrics | Hit Rate@{k}: {hit_rate*100:.2f}%"
    )  # Provenance log
    print(f"\nHit Rate@{k} : {hit_rate*100:.2f}%")


if __name__ == "__main__":
    main()
