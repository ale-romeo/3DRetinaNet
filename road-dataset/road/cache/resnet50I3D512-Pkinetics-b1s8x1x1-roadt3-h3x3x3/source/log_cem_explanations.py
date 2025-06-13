import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def log_cem_explanations(concept_preds_bin, ego_preds, output_dir="cem_outputs", prefix="val"):
    """
    Salva le predizioni di concetti attivi e ego_action in un file JSON e una heatmap PNG.

    Parameters:
    - concept_preds_bin: np.ndarray di shape [N, K] (binaria, concetti attivi)
    - ego_preds: np.ndarray di shape [N, C] (logits o probabilità per ogni ego_action)
    - output_dir: directory dove salvare i file
    - prefix: prefisso per i nomi dei file
    """
    os.makedirs(output_dir, exist_ok=True)

    explanations = []
    for i in range(len(ego_preds)):
        pred_action = int(np.argmax(ego_preds[i]))
        active_concepts = np.where(concept_preds_bin[i] == 1)[0].tolist()
        explanations.append({
            "frame_id": i,
            "pred_ego_action": pred_action,
            "active_concepts": active_concepts
        })

    json_path = os.path.join(output_dir, f"{prefix}_explanations.json")
    with open(json_path, "w") as f:
        json.dump(explanations, f, indent=2)

    # Heatmap dei concetti attivi
    plt.figure(figsize=(12, 6))
    sns.heatmap(concept_preds_bin.astype(int), cmap="YlGnBu", cbar=True)
    plt.xlabel("Concept ID")
    plt.ylabel("Frame ID")
    plt.title("Concetti attivi nel tempo")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"{prefix}_concepts_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    return json_path, heatmap_path


# Esempio di utilizzo:
# concept_preds_bin = (concept_preds_all > 0.5).astype(int)
# ego_preds = ...  # logits o sigmoid
# log_cem_explanations(concept_preds_bin, ego_preds)

