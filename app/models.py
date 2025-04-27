import os
import time
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path
from app.logger import logger


class Base:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        # Ensures singleton pattern
        if cls not in cls._instances:
            cls._instances[cls] = super(Base, cls).__new__(cls)
        return cls._instances[cls]

    def __init__(self, *args, **kwargs):
        self.saved_model_path = None
        self.dataset_path = None
        self.model = None
        self.df = None

    def load_model(self):
        with open(self.saved_model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def load_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def recommend(self, user_id, n=20):
        raise NotImplementedError


class SVDRecommender(Base):
    def __init__(self, model_path=None):
        super().__init__()
        root = (
            Path(__file__).resolve().parent.parent
        )  # Resolves to Movie-Recommendation/
        if model_path:
            self.saved_model_path = root / model_path
        else:
            self.saved_model_path = root / "trained_models" / "trained_model.pkl"
        self.dataset_path = root / "data" / "final_processed_data.csv"
        self.model = self.load_model()
        self.df = self.load_data()

    def load_data(self):
        df_all = pd.read_csv(self.dataset_path)
        df_all.rename(
            columns={
                "User_ID": "user_id",
                "Movie_Name": "movie_id",
                "Rating": "rating",
            },
            inplace=True,
        )

        df_all["user_id"] = df_all["user_id"].astype(int)

        df_all["rating"] = pd.to_numeric(df_all["rating"], errors="coerce")

        movie_avg = df_all.groupby("movie_id")["rating"].transform("mean")
        df_all["rating"] = df_all["rating"].fillna(movie_avg)

        global_avg = df_all["rating"].mean()
        df_all["rating"] = df_all["rating"].fillna(global_avg)

        logger.info(f"Loaded {len(df_all)} rows from {self.dataset_path}.")
        return df_all

    def stratified_cross_validate(
        self, model_class, df, n_splits=5, reader=Reader(rating_scale=(1, 5))
    ):
        y_discrete = df["rating"].round().astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = df.index.values

        rmses = []
        maes = []
        fit_times = []
        test_times = []

        for train_index, test_index in skf.split(X, y_discrete):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            train_data = Dataset.load_from_df(
                train_df[["user_id", "movie_id", "rating"]], reader
            )
            trainset = train_data.build_full_trainset()

            model = model_class()

            # Measure fit time for this fold
            fold_fit_start = time.time()
            model.fit(trainset)
            fold_fit_time = time.time() - fold_fit_start
            fit_times.append(fold_fit_time)

            # Measure test time for this fold and compute predictions
            fold_test_start = time.time()
            test_preds = []
            for _, row in test_df.iterrows():
                uid = row["user_id"]
                iid = row["movie_id"]
                true_rating = row["rating"]
                pred = model.predict(uid, iid).est
                test_preds.append((true_rating, pred))
            fold_test_time = time.time() - fold_test_start
            test_times.append(fold_test_time)

            # Compute RMSE and MAE
            se = [(true - pred) ** 2 for true, pred in test_preds]
            ae = [abs(true - pred) for true, pred in test_preds]
            rmse = np.sqrt(np.mean(se))
            mae = np.mean(ae)
            rmses.append(rmse)
            maes.append(mae)

        mean_rmse = np.mean(rmses)
        mean_mae = np.mean(maes)
        mean_fit_time = np.mean(fit_times)
        mean_test_time = np.mean(test_times)
        return mean_rmse, mean_mae, mean_fit_time, mean_test_time

    def save_model(self):
        with open(self.saved_model_path, "wb") as f:
            pickle.dump(self.model, f)

    def train(self):
        reader = Reader(rating_scale=(1, 5))

        # Provenance: Start tracking training time
        training_start_time = time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info("Performing stratified cross-validation with 5 folds...")
        (
            mean_rmse,
            mean_mae,
            mean_fit_time,
            mean_test_time,
        ) = self.stratified_cross_validate(SVD, self.df, n_splits=5, reader=reader)
        logger.info(f"RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")
        logger.info(
            f"Average Fit Time per fold: {mean_fit_time:.4f} sec, Average Test Time per fold: {mean_test_time:.4f} sec"
        )

        start_time = time.time()
        data = Dataset.load_from_df(self.df[["user_id", "movie_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        model = SVD()
        model.fit(trainset)
        training_time = time.time() - start_time
        logger.info(f"Training time (full training set): {training_time:.4f} seconds")

        self.save_model()

        model_size = os.path.getsize(self.saved_model_path) / 1024.0
        logger.info(
            f"Model saved as {self.saved_model_path}, size: {model_size:.2f} KB"
        )
        logger.info("Model training complete.")

        # Provenance: Log metadata
        model_version = time.strftime("%Y%m%d_%H%M%S")  # timestamp version
        pipeline_version = "v1.0"  # static for now, can be dynamic

        logger.info("Provenance Info:")
        logger.info(f"  Model Version: {model_version}")
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Pipeline Version: {pipeline_version}")
        logger.info(f"  Training Start: {training_start_time}")
        logger.info(f"  Training End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")
        logger.info(f"  Model Size: {model_size:.2f} KB")

        return model, self.df, training_time, model_size

    def recommend(self, user_id, n=20):
        all_movies = self.df["movie_id"].unique()
        watched_movies = self.df[self.df["user_id"] == user_id]["movie_id"].tolist()

        inference_start = time.time()
        predictions = [
            self.model.predict(user_id, movie)
            for movie in all_movies
            if movie not in watched_movies
        ]
        predictions.sort(key=lambda x: x.est, reverse=True)
        recommended = [pred.iid for pred in predictions[:n]]
        inference_time = time.time() - inference_start
        logger.info(f"Inference time for user {user_id}: {inference_time:.4f} seconds")
        return recommended, inference_time
