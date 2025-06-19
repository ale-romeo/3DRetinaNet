import json
import torch
from tqdm import tqdm

# === CONFIG ===
ANNOTATION_FILE = 'road_trainval_v1.0.json'
FRAME_TO_TRIPLETS_FILE = 'frame_to_triplets.json'
OUTPUT_MATRIX_FILE = 'concept_matrix.pt'
OUTPUT_INDEX_FILE = 'frame_index.json'

# === LOAD LABELS (solo quelli rilevanti) ===
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)
triplet_labels = data['triplet_labels']  # <- usa solo questi 68
triplet_to_index = {label: idx for idx, label in enumerate(triplet_labels)}

# === LOAD FRAME TRIPLET MAPPING ===
with open(FRAME_TO_TRIPLETS_FILE, 'r') as f:
    frame_to_triplets = json.load(f)

# === BUILD MATRIX ===
frame_keys = list(frame_to_triplets.keys())
num_frames = len(frame_keys)
num_triplets = len(triplet_labels)

concept_matrix = torch.zeros((num_frames, num_triplets), dtype=torch.float32)

print(f"Building concept matrix: {num_frames} frames × {num_triplets} filtered triplets")

for i, frame_id in tqdm(enumerate(frame_keys), total=num_frames):
    for triplet in frame_to_triplets[frame_id]:
        idx = triplet_to_index.get(triplet)
        if idx is not None:
            concept_matrix[i, idx] = 1.0
        else:
            print(f"[!] Triplet '{triplet}' non trovato in triplet_labels")

print(f"[✓] Matrice concetti costruita: {concept_matrix.sum().item()} triplets attivi su {num_frames * num_triplets} totali")

# === SAVE ===
torch.save(concept_matrix, OUTPUT_MATRIX_FILE)
with open(OUTPUT_INDEX_FILE, 'w') as f:
    json.dump(frame_keys, f)

print(f"[✓] Salvato: {OUTPUT_MATRIX_FILE}  ({concept_matrix.shape})")
print(f"[✓] Salvato: {OUTPUT_INDEX_FILE}  ({len(frame_keys)} frame keys)")
