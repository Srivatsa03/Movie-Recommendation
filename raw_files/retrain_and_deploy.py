import shutil
import datetime
import subprocess
from train import train_model

if __name__ == "__main__":
    model, df_all, training_time, model_size = train_model(
        "data/final_processed_data.csv"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"svdpp_model_{timestamp}.pkl"
    version_path = f"models/{version_name}"

    shutil.copyfile("trained_models/trained_model.pkl", version_path)

    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        commit_hash = "unknown"

    with open("model_versions.log", "a") as f:
        f.write(f"{timestamp},{version_name},git={commit_hash}\n")

    print(f" Retraining complete. Model saved as: {version_path}")
