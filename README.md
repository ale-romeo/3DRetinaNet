# 3D-RetinaNet with Concept-Based Explainability for ROAD dataset

This repository contains code for an enhanced version of 3D-RetinaNet, originally proposed with the [ROAD dataset](https://github.com/gurkirt/road-dataset). We extend the model with **concept-based explainability**, integrating a Concept Embedding Module (CEM) to make the architecture explainable by design. This work is described in more detail in our [project paper (preprint)](https://drive.google.com/file/d/1TzWC5a-9tpNwmuRlWWYQ18PCFQnOgRrW/view?usp=sharing).

## Table of Contents

- Requirements
- Training 3D-RetinaNet
- Testing and Building Tubes
- Performance
- Concept-Based Explainability Extension (CEM)
- Citation
- Reference

## Requirements

We need three things to get started with training: datasets, kinetics pre-trained weight, and pytorch with torchvision and tensoboardX.

### Dataset download and pre-processing

We used only the [ROAD dataset](https://github.com/gurkirt/road-dataset), introduced in the [dataset release paper](https://arxiv.org/pdf/2102.11585.pdf).

### Pytorch and weights

- Install [Pytorch](https://pytorch.org/) and [torchvision](http://pytorch.org/docs/torchvision/datasets.html)
- Install tensorboardX via `pip install tensorboardx`
- Download Kinetics-400 pretrained weights into `kinetics-pt/`, using [this script](./kinetics-pt/get_kinetics_weights.sh) or [Google Drive](https://drive.google.com/drive/folders/1xERCC1wa1pgcDtrZxPgDKteIQLkLByPS?usp=sharing)

## Training 3D-RetinaNet

You will need 4 GPUs (each with at least 10GB VRAM). Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ \
  --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 \
  --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```

## Testing and Building Tubes

For evaluation and tube generation, use:

```bash
python main.py /home/user/ /home/user/ /home/user/kinetics-pt/ \
  --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 \
  --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```

## Performance

Results obtained after 60 training epochs with the explainable model (CEM enabled):

- **Agentness** MEANAP: **54.67**
- **Agent** MEANAP: **37.10**
- **Action** MEANAP: **23.03**
- **Location** MEANAP: **27.92**
- **Duplex** MEANAP: **26.09**
- **Triplet** MEANAP: **19.49**
- **Ego-action** MEANAP: **42.40**

[CEM] Concept Prediction:

- Accuracy: **77.05%**
- F1 Micro: **0.2030**, F1 Macro: **0.1060**

## 🔍 Concept-Based Explainability Extension (CEM)

We extended 3D-RetinaNet to support **explainability by design** through integration of a **Concept Embedding Module (CEM)**.

### Key Changes:

- 🧠 **CEM Head**: Learns interpretable concept embeddings with dual embeddings (active/inactive).
- 🔁 **Transformer Encoder**: Encodes temporal patterns on the concept bottleneck.
- 🎯 **Ego Head**: Replaced with a concept-driven prediction head.
- 📊 **Concept Supervision**: Via BCEWithLogitsLoss using dynamic class balancing.
- 🧩 **Triplet Annotations**: Dataset modified to include triplet-based concept labels.
- 📉 **Loss Tracking**: Training logs `cem_loss` alongside detection losses.
- 📈 **Evaluation Enhancements**: F1-scores, per-concept stats, hardest concept analysis.

### Enable CEM with:

```bash
--USE_CEM=True --num_concepts=68 --cem_dim=16
```

## Citation

If this work was helpful, consider citing the original 3D-RetinaNet paper:

```bibtex
@ARTICLE{singh2022road,
  author = {Singh, Gurkirt and others},
  journal = {IEEE TPAMI},
  title = {ROAD: The ROad event Awareness Dataset for autonomous Driving},
  year = {5555},
  doi = {10.1109/TPAMI.2022.3150906},
}
```

## Reference

- [ROAD dataset](https://github.com/gurkirt/road-dataset)
- [Original 3D-RetinaNet TPAMI paper](https://www.computer.org/csdl/journal/tp/5555/01/09712346/1AZL0P4dL1e)
- [Our concept-based explainability paper (preprint)](https://drive.google.com/file/d/1TzWC5a-9tpNwmuRlWWYQ18PCFQnOgRrW/view?usp=sharing)

