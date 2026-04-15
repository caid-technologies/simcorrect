# CadIntel — CAD AI/ML Printability Assistant

A two-stage AI/ML pipeline that analyses CAD drawings for 3D printability. It classifies drawings as **PASS / REVIEW / FAIL** (POI1), diagnoses the specific defect type (POI2), suggests actionable fixes, and presents everything in an interactive Streamlit demo with annotated overlays and repaired outputs.

---

## Project Structure

```
cad_intel/
├── code/                          # POI1 training pipeline
│   ├── poi2/                      # POI2 labeling + training scripts
│   ├── dxf_feature_extractor.py
│   ├── dxf_to_mesh.py
│   ├── enrich_labels.py
│   ├── label_from_rules_v2.py
│   ├── make_labels.py
│   ├── model_def.py
│   ├── poi1_dataloader.py
│   ├── poi1_extra_checks_with_standards.py
│   ├── read_meta.py
│   ├── rules_loader.py
│   ├── sanity_check.sh
│   ├── split.py
│   ├── test_pipeline_one.py
│   ├── train_master.py
│   ├── train_master_progress.py
│   └── validate_pairs.py
├── dataset/
│   ├── triview_20K/               # 27K CAD samples (images + DXF)
│   └── labels.csv
├── rules/
│   ├── triview_rules_v1_0.yaml    # POI1 ruleset (7 manufacturing profiles)
│   ├── defects_po2_v1.yaml        # POI2 defect category rules
│   └── fix_map.yaml               # Rule → fix action mapping
├── runs/                          # Training checkpoints
│   ├── exp_001/
│   ├── master_v1/                 # Production POI1 model
│   └── sanity_small/
├── output/                        # Inference outputs
│   ├── poi2_model/                # Trained POI2 model
│   ├── poi2_data/
│   ├── overlays_annot/
│   ├── overlays_annot_rules/
│   └── [CSVs and reports]
├── demo_bundle_v2/                # Final demo package
│   ├── app/app.py                 # Streamlit demo UI
│   ├── clean_repair_visuals.py
│   ├── repair_pretty.py
│   ├── strict_clean_and_snap.py
│   ├── inputs/                    # 3 demo sample images
│   ├── repaired/                  # Repaired CAD outputs
│   ├── overlays/                  # Annotated overlays
│   ├── previews2d/                # Side-by-side pair renders
│   ├── poi2/                      # Per-sample recommendation CSVs
│   └── metrics/                   # poi2_confusion.csv, poi2_eval.json
├── config.yaml                    # Base training config
├── config_master.yaml             # Production POI1 config
├── config_small.yaml              # Quick sanity-check config (2 epochs)
└── config_poi2.yaml               # POI2 training config
```

---

## Pipeline Overview

```
                        27K CAD samples (images + DXF)
                                    │
                    ┌───────────────▼───────────────┐
                    │           POI1                │
                    │   Rules engine + ML model     │
                    │   Output: PASS / REVIEW / FAIL│
                    └───────────────┬───────────────┘
                                    │ REVIEW + FAIL only
                                    │ (poi1_to_handoff.py)
                    ┌───────────────▼───────────────┐
                    │           POI2                │
                    │   Defect classification       │
                    │   Output: defect category     │
                    │   + fix recommendations       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │       Streamlit Demo          │
                    │  Input │ Repaired │ Overlay   │
                    │  POI2 recs │ Metrics          │
                    └───────────────────────────────┘
```

---

## Stage 1 — POI1: Printability Classification

**Question: Is this drawing printable?**
**Classes: PASS / REVIEW / FAIL**
**Dataset: All 27,000 samples**

### Model Architecture

```
Image (PNG/JPG)
  └─► MobileNetV3-Small ──► 256-dim embedding
                                      │
DXF geometry (512 tokens × 8 dim)    ├─► Concat ──► 512-dim ──► Head ──► 3 classes
  └─► MLP (8→128→256) ──► mean ──► 256-dim
```

**DXF token format** — 8 dimensions per entity:
`[type, x1, y1, x2, y2, radius, angle_start, angle_end]`
Types: LINE=1, CIRCLE=2, ARC=3, LWPOLYLINE=4. All coordinates normalised to [0,1].

### Configs

| Config | epochs | batch | Use |
|--------|--------|-------|-----|
| `config_small.yaml` | 2 | 32 | Sanity check |
| `config.yaml` | 10 | 32 | Standard run |
| `config_master.yaml` | 1 | 8 | Production (full pipeline, MPS) |

`config_master.yaml` also loads the ruleset directly:
```yaml
ruleset_path: rules/triview_rules_v1_0.yaml
dxf_max_tokens: 512
mixed_precision: true
device: mps
```

### Dataset Split

~27,000 samples, deterministic SHA-256 hash split:

| Split | Count |
|-------|-------|
| train | ~21,600 |
| val | ~2,700 |
| test | ~2,700 |

### POI1 Scripts (code/)

| Script | Purpose |
|--------|---------|
| `read_meta.py` | Inspect first 5 entries of metadata.json |
| `validate_pairs.py` | Check image ↔ DXF pairing; writes pairs.json |
| `split.py` | Deterministic 80/10/10 split via SHA-256 |
| `make_labels.py` | Generate labels from metadata.json fields |
| `dxf_feature_extractor.py` | Extract geometric features from DXF files |
| `label_from_rules_v2.py` | Apply YAML rules → labels_rules.csv |
| `enrich_labels.py` | Merge rule_hits into labels_enriched.csv |
| `rules_loader.py` | Load and validate YAML ruleset |
| `poi1_dataloader.py` | PyTorch Dataset: images + DXF tokens + labels |
| `model_def.py` | Lightweight CNN+MLP model (used for dry-run) |
| `train_master_progress.py` | Training with step-level logging (dry-run) |
| `train_master.py` | Production training: MPS, mixed precision, best-model checkpointing |
| `poi1_extra_checks_with_standards.py` | Per-sample checks vs engineering standards |
| `dxf_to_mesh.py` | Convert DXF profiles to 3D STL via Shapely + Trimesh |
| `sanity_check.sh` | Full pre-flight checklist before training |
| `test_pipeline_one.py` | End-to-end result lookup for a single sample ID |

### POI1 Ruleset (triview_rules_v1_0.yaml)

7 manufacturing profiles: `MP` (machined), `SM` (sheet metal), `IM` (injection moulding), `AP-FDM`, `AP-SLA`, `AP-SLS`, `AP-SLM`

Key rules:

| Rule ID | Description | Severity |
|---------|-------------|----------|
| A1_parse_ok | DXF must parse without error | critical |
| A2_no_degenerate | No zero-length segments | critical |
| A5_closed_loops | Closed profiles must close within tolerance | major |
| D1_wall_min | Minimum wall thickness | major |
| D2_hole_min | Minimum hole diameter | major |
| E1_tiny_arc_limit | Arc radii above minimum | minor |
| E2_density_tail | Entity density not extreme | major |
| G_AP_overhang | Overhang within profile max | major |

Scoring policy: any critical → FAIL; score ≥ 2 → FAIL; score = 1 → REVIEW; score = 0 → PASS

### Engineering Standards Checked (poi1_extra_checks_with_standards.py)

| Check | Standard | Threshold |
|-------|----------|-----------|
| Units | ISO 10303 / ASME Y14.5 | Must be mm |
| Scale | ISO 286-1 | 0.95–1.05 |
| Tolerance | ASTM F2921-11 | 0.05–0.50 mm |
| Min clearance | ASTM F2921-11 / Stratasys | ≥ 0.20 mm |

---

## Stage 2 — POI2: Defect Classification

**Question: What is wrong with it?**
**Classes: 8 defect categories**
**Dataset: REVIEW + FAIL samples from POI1 only**

### Defect Categories (defects_po2_v1.yaml)

| Category | Triggered by |
|----------|-------------|
| OVERHANG_SUPPORT | max_overhang_deg > 45° |
| THIN_WALL | min_wall_mm < max(0.80mm, 3× line width) |
| MIN_FEATURE | hole/slot/emboss < 0.50mm |
| CLEARANCE | xy_clearance < 0.20mm or z_clearance < layer height |
| TOLERANCE | tol_band outside 0.05–0.50mm |
| THERMAL_WARPING | flat area > 25cm² or warp_risk_flag |
| LAYER_SHIFT_STRINGING | travel > 120mm p95, tower aspect > 10, stringing flag |
| OTHER | fallback |

### Model

ResNet-18 fine-tuned from ImageNet weights, image-only (no DXF), `OTHER` class excluded from training.

### Config (config_poi2.yaml)

```yaml
manifest: output/poi2_manifest.csv
save_dir: output/poi2_runs
device: mps
img_size: 224
batch_size: 64
epochs: 5
lr: 1e-3
val_split: 0.1
test_split: 0.1
seed: 42
```

### Fix Recommendations (fix_map.yaml)

Each rule ID maps to concrete fix actions:

| Rule | Fix Action |
|------|-----------|
| D2_hole_min | enlarge_holes_to: 12mm |
| O1_overhang_max | add_supports (tree, 45°) or add_gusset |
| W1_thin_wall | thicken_wall_to: 1.2mm |
| C1_clearance_min | increase_clearance_to: 0.3mm |
| B1_bridge_len_max | add_ribs every 20mm |
| S1_slot_width_min | widen_slot_to: 0.6mm |
| F1_text_size_min | enlarge_text: height ≥ 3mm |

### POI2 Scripts (code/poi2/)

| Script | Purpose |
|--------|---------|
| `poi1_to_handoff.py` | Filter REVIEW/FAIL from POI1 → handoff.jsonl with checksums |
| `build_labels_po2_v2.py` | Map rule hits + POI1 checks → POI2 defect labels |
| `poi2_train.py` | Train ResNet-18 defect classifier |

---

## Demo App (demo_bundle_v2/)

A Streamlit app presenting the full integrated pipeline on 3 curated samples (IDs: 100041, 100063, 100129).

**Run:**
```bash
cd demo_bundle_v2
streamlit run app/app.py
```

**Shows per sample:**
- Input image vs repaired image
- Rule/prediction overlay
- Side-by-side 2D preview
- POI2 recommendations table
- POI2 confusion matrix + eval JSON (downloadable)

### Demo Visual Scripts

| Script | Purpose |
|--------|---------|
| `strict_clean_and_snap.py` | Clean repaired renders, snap support structures into position, generate side-by-side pair previews |
| `repair_pretty.py` | Prettify and annotate repair suggestion visuals |
| `clean_repair_visuals.py` | Clean up overlay/repair output images |
| `bl_render_pair_2d.py` | Blender script: render 2D tri-view pairs |
| `bl_render_svg_3d.py` | Blender script: render 3D SVG views |

---

## Running the Full Pipeline

### 1. Validate dataset
```bash
python code/validate_pairs.py --root dataset/triview_20K
```

### 2. Generate labels
```bash
python code/label_from_rules_v2.py \
  --root    dataset/triview_20K \
  --ruleset rules/triview_rules_v1_0.yaml \
  --out     dataset/triview_20K/labels_rules.csv

python code/enrich_labels.py
```

### 3. Split dataset
```bash
python code/split.py \
  --root    dataset/triview_20K \
  --labels  labels.csv \
  --outdir  dataset/triview_20K
```

### 4. Sanity check
```bash
bash code/sanity_check.sh \
  dataset/triview_20K \
  dataset/triview_20K/labels.csv \
  dataset/triview_20K
```

### 5. Train POI1
```bash
python code/train_master.py \
  --config config_master.yaml \
  --save   runs/master_v1
```

### 6. POI1 → POI2 handoff
```bash
python code/poi2/poi1_to_handoff.py \
  output/poi1_extra_checks.csv \
  --images_dir dataset/triview_20K/images \
  --dxf_dir    dataset/triview_20K/dxf \
  --out_dir    output/poi2_data

python code/poi2/build_labels_po2_v2.py \
  --handoff     output/poi2_data/handoff.jsonl \
  --enriched_csv dataset/triview_20K/labels_enriched.csv \
  --poi1_checks  output/poi1_extra_checks.csv \
  --out_csv      output/poi2_data/labels_po2.csv
```

### 7. Train POI2
```bash
python code/poi2/poi2_train.py \
  --labels     output/poi2_data/labels_po2.csv \
  --images_dir dataset/triview_20K/images \
  --out_dir    output/poi2_runs
```

### 8. Run demo
```bash
cd demo_bundle_v2
streamlit run app/app.py
```

---

## Environment Setup

The project requires a dedicated conda environment. The `base` environment does **not** have the necessary packages (torch, ezdxf, streamlit etc. are absent from base).

### Create environment from scratch

```bash
# Create and activate
conda create -n cadintel python=3.10
conda activate cadintel

# PyTorch for Apple Silicon (MPS)
conda install pytorch torchvision -c pytorch

# Core pipeline
pip install ezdxf pyyaml numpy pillow pandas

# Demo app
pip install streamlit

# DXF → mesh conversion (dxf_to_mesh.py only)
pip install shapely trimesh

# Misc utilities
pip install matplotlib tqdm
```

### Verify MPS (Apple Silicon GPU) is available

```bash
python - <<'PY'
import torch
print("MPS available:", torch.backends.mps.is_available())
PY
```

### Notes

- Requires Python 3.10 and PyTorch 2.x
- Runs on Apple Silicon (MPS) automatically, falls back to CPU if unavailable
- Two conda installs are present on this machine (`/Users/apple/miniforge3` and `/opt/homebrew/anaconda3`) — always use miniforge3's conda to avoid conflicts:
  ```bash
  /Users/apple/miniforge3/bin/conda activate cadintel
  ```

---

## Results

- POI1 best val accuracy: tracked in `runs/master_v1/val_report.json`
- POI2 confidence: ~0.74 (from presentation)
- Demo samples: 100041 (FAIL), 100063, 100129
- Full eval metrics: `demo_bundle_v2/metrics/poi2_eval.json`
