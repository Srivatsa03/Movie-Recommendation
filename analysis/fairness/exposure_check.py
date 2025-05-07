

import json, collections, math, pandas as pd, sys, glob

# --- config -----------------------------------------------------------
LOG_PATTERN = "logs/recs_2025-05-06*.jsonl"   # adjust date / path as needed
DATA_PATH   = "data/final_processed_data.csv"
LIGHT_RATING_THRESHOLD = 5
# ----------------------------------------------------------------------

# 1  load rating counts per user to label light users
ratings      = pd.read_csv(DATA_PATH)
rating_count = ratings.groupby("User_ID").size().to_dict()
is_light     = lambda uid: rating_count.get(uid, 0) <= LIGHT_RATING_THRESHOLD

# 2  gather genre counts
all_genres   = collections.Counter()
light_genres = collections.Counter()

for path in glob.glob(LOG_PATTERN):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec  = json.loads(line)
            uid  = rec["user_id"]
            for g in rec["genres"]:
                all_genres[g] += 1
                if is_light(uid):
                    light_genres[g] += 1

# convert to probability vectors
def to_probs(counter):
    total = sum(counter.values())
    return {g: c / total for g, c in counter.items()}

P_all   = to_probs(all_genres)
P_light = to_probs(light_genres)

# 3  KL divergence
kl = 0.0
for g, p in P_light.items():
    q = P_all.get(g, 1e-12)      # smoothing for missing genres
    kl += p * math.log(p / q)

print(f"KL(light ‖ global) = {kl:.3f}")
