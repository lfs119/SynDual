# SynDual

**SynDual** is a dual-pocket-conditioned molecular generation framework supported by a newly constructed dataset linking protein–ligand structures, residue-level interactions, molecular products and synthesis pathways. 



---

## 1. Download data expected by SynDual

The complete training and evaluation datasets (comprising Set1 and Set2, as well as their independent train, validation, and test splits) can be accessed and downloaded via DOI [here](https://doi.org/10.5281/zenodo.23997651).  

Expected files include:

```text
set2_train_molecules.csv
set2_val_molecules.csv
set2_test_molecules.csv
set2_train_product_routes.jsonl
set2_val_product_routes.jsonl
set2_test_product_routes.jsonl
```

Reference clean/split statistics:

- Set1: 556 unique ligands, 2,626 synthesis-aware rows, 2,419 dual-target records, 582 route tasks.
- Set1 train/val/test ligands: 444 / 56 / 56.
- Set2: 66,944 product-representation rows, 66,924 unique chemical identities, 70,293 route tasks.
- Set2 train/val/test route tasks: 63,245 / 3,523 / 3,525.
- Set2 train/val/test unique identities: 60,250 / 3,337 / 3,337.
- Set1 scaffold overlap: 0.
- Set2 chemical-identity overlap: 0.
- Set1/Set2 exact InChIKey and canonical-SMILES overlap: 0.



---


## 2. Installation

```bash
cd SynDual
mamba env create -f environment.yml
conda activate syndual
pip install -e .
```

or, in an existing compatible CUDA/PyTorch environment:

```bash
pip install -r requirements.txt
pip install -e .
```

The first IBM checkpoint load requires internet access unless the files are already cached.  Model weights used in this study (including the Stage 2 molecule generation model, synthesis route generation model, and joint molecule-route generation model, alongside the Stage 3 pocket adapter), training logs, and detailed evaluation results on the validation and test sets are available [here](https://doi.org/10.5281/zenodo.23996471)

---


## 2. Stage 2: Set2 molecular adaptation

```bash
python scripts/train_set2_smiles.py \
    --config configs/stage1_set2_smiles.yaml
```

Best checkpoint:

```text
runs/stage1_set2_smiles/best/
├── backbone/
├── tokenizer/
└── trainer_state.json
```

---

## 3. Stage 2: route decoder

```bash
python scripts/train_route_decoder.py \
    --config configs/stage2_route.yaml
```

Default behavior:

- Stage-1 GP-MoLFormer is frozen;
- frozen backbone is kept in `eval()` mode so dropout does not perturb route memory;
- complete v4 route sequence is learned with `loss_mode=all_tokens`;
- no route truncation is permitted.

Best `model_syn` checkpoint:

```text
runs/stage2_route/best/
├── backbone/
├── product_tokenizer/
├── route_tokenizer/
├── route_decoder.pt
└── syndual_config.json
```

---

## 4. Stage 2: optional joint fine-tuning

```bash
python scripts/joint_finetune_model_syn.py \
    --config configs/stage2_joint.yaml
```


The backbone uses a much smaller learning rate than the route decoder.

---

## 5. Evaluation

```bash
python scripts/evaluate_model_syn_stage2e.py \
    --config configs/eval_model_syn_stage2E.yaml  --generation-mode full  --output_dir  path/to/...
```

Outputs include:

- teacher-forced route loss and perplexity;
- route grammar-valid fraction;
- all-generated-molecules-valid fraction;
- final route product vs conditioning-product identity match;
- exact serialized-route match;
- decoder/lexicalization error fraction;
- unconditional SMILES validity and uniqueness.

Exact route match is intentionally reported as a strict reconstruction metric, not as the only criterion for synthesis quality, because multiple valid synthesis plans may exist for one product.

---

## 6. End-to-end inference

```bash
python scripts/generate_smiles_and_route_stage2e.py \
    --config configs/generate_smiles_and_route_stage2E.yaml
```

Pipeline:

```text
GP-MoLFormer
     |
     v
generated SMILES
     |
     v
GP-MoLFormer product hidden states
     |
     v
lightweight route decoder
     |
     v
lexicalized route IDs
     |
     v
source-level SynLaD route tokens
```

Each JSONL output stores the generated molecule, source-level route token list, grammar validity, molecular-SMILES validity and whether the route's final product matches the generated conditioning molecule.

---

## 7. Stage 3: pocket-conditioned PH4 generation

Stage 3 extends the molecular generator with pocket-local pharmacophore (PH4) conditioning and synthesis-route generation. Before training, prepare the local PH4 target file and update the data/checkpoint paths in the configuration files for your environment.

### 7.1 Train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage3_pocket_ph4_v2.py \
    --config configs/stage3_pocket_ph4_v2.yaml
```

Training settings and input/output paths are defined in `configs/stage3_pocket_ph4_v2.yaml`.

### 7.2 Test and generate

Run generation on the test split:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_stage3_pocket_and_route_ph4_v2.py \
    --config configs/eval_model_syn_stage3_ph4_v2_v24.yaml \
    --split test \
    --num-molecules 16 \
    --output-dir path/to/outputs/stage3_ph4_v2_generation/test
```

The generated molecules and their synthesis-route predictions are saved under the directory specified by `--output-dir`.
---

## License

This software is released under a custom license.

Academic use of this software is free and does not require any permission.
We encourage academic users to cite our research paper (if applicable).

For commercial use, please contact us for permission.