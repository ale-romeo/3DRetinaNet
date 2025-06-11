import json
from tqdm import tqdm
from collections import defaultdict
import json

# ====== Config ======
json_path = "road_trainval_v1.0.json"  # 🔁 aggiorna path
output_path = "frame_to_triplets.json"
# ====================

with open(json_path, "r") as f:
    data = json.load(f)

db = data["db"]
all_triplet_labels = data["all_triplet_labels"]

frame_to_triplets = {}

for video_id, video_data in db.items():
    triplet_tubes = video_data.get("triplet_tubes", {})
    for tube_id, tube in triplet_tubes.items():
        triplet_idx = tube["label_id"]
        triplet_str = all_triplet_labels[triplet_idx]

        for frame_str in tube["annos"].keys():
            frame_key = f"{video_id}_{int(frame_str):05d}"
            frame_to_triplets.setdefault(frame_key, []).append(triplet_str)

# Controlla il risultato
for k, v in list(frame_to_triplets.items())[:5]:
    print(k, "→", v)

# Salva il file
with open(output_path, "w") as f:
    json.dump(frame_to_triplets, f, indent=2)

print(f"[✓] Salvato frame_to_triplets.json con {len(frame_to_triplets)} chiavi.")
