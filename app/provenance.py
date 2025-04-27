import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Set the directory where provenance logs will be saved
PROVENANCE_LOG_DIR = Path(__file__).resolve().parent.parent / "provenance_logs"
PROVENANCE_LOG_DIR.mkdir(exist_ok=True)  # Ensure the directory exists


def get_git_commit_hash():
    """
    Retrieves the current Git commit hash to track the version of the pipeline/code.
    Prioritizes the environment variable GIT_COMMIT_HASH if set (e.g., via Docker or Jenkins).

    Returns:
        str: The Git commit hash (short form) or 'unknown'.
    """
    # Check for environment variable (injected during Docker/Jenkins build)
    commit_hash = os.getenv("GIT_COMMIT_HASH")
    if commit_hash and commit_hash != "unknown":
        return commit_hash

    # Fallback to local git repo (if running locally)
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except Exception:
        return "unknown"


def log_provenance(user_id, model_version, data_version, recommendations):
    """
    Logs the provenance (history) for each recommendation made.

    Args:
        user_id (int): The user ID for whom recommendations were generated.
        model_version (str): The version or filename of the model used (e.g., 'trained_model.pkl').
        data_version (str): The version or filename of the data used (e.g., 'final_processed_data.csv').
        recommendations (list): The list of recommended movie IDs/names.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),  # When the recommendation was made (UTC)
        "user_id": user_id,  # For which user
        "model_version": model_version,  # Which model (e.g., 'trained_model.pkl')
        "data_version": data_version,  # Which data (e.g., 'final_processed_data.csv')
        "pipeline_commit": get_git_commit_hash(),  # Which code version (Git commit hash)
        "recommendations": recommendations,  # The actual recommendations made
    }

    # Save the log entry as a JSON line (newline-delimited JSON for easy parsing)
    log_file = PROVENANCE_LOG_DIR / "provenance_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(
        f"[Provenance] Logged for user {user_id} with commit {log_entry['pipeline_commit']}"
    )
