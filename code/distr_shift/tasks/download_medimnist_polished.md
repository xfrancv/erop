# Plan

Extend the real-data experiments with four additional datasets from the
MedMNIST v2 collection: **PathMNIST**, **OCTMNIST**, **TissueMNIST** and
**OrganAMNIST**.

# Context

The real-data pipeline currently uses three MedMNIST datasets (DermaMNIST,
BloodMNIST, RetinaMNIST). The involved scripts are `download_datasets.py`,
`analyze_datasets.py` and `run_base_predictor_exp.py`; the per-dataset metadata
lives in `data_tools/registry.py` and the loaders in `data_tools/loaders.py`.

The MedMNIST homepage is https://medmnist.com/. All MedMNIST v2 archives are
28×28 `.npz` files on the same Zenodo record already used by the existing three
datasets (`_ZENODO = "https://zenodo.org/records/10519652/files/"`), each
carrying separate `train`/`val`/`test` arrays. They are consumed by the existing
`medmnist` loader (`data_tools/loaders.py::_load_medmnist`) with no new loader
code.

# Task

## 1. Datasets to add

Add four entries to `DATASETS` in `data_tools/registry.py`. Official split sizes
and image type (used to justify the split policy in §2):

| key           | classes | channels | train  | val    | test   |
|---------------|---------|----------|--------|--------|--------|
| `pathmnist`   | 9       | RGB      | 89,996 | 10,004 | 7,180  |
| `octmnist`    | 4       | gray     | 97,477 | 10,832 | 1,000  |
| `tissuemnist` | 8       | gray     | 165,466| 23,640 | 47,280 |
| `organamnist` | 11      | gray     | 34,581 | 6,491  | 17,778 |

Download uses the existing pattern, e.g.
`files=[(_ZENODO + "pathmnist.npz?download=1", "pathmnist.npz")]`.

## 2. Split policy: a per-dataset `val_role` field

Different datasets need different roles for the official `val` split, so encode
the policy per-dataset in the registry rather than as a CLI flag that can be
forgotten. Add a field to `DatasetSpec`:

```python
val_role: str = "test"   # "test" | "train"
```

- **`val_role="test"` (default, current behaviour).** The official `val` split
  is merged into the *test* subset (used only for adaptation/evaluation, never
  for training or calibration); the model-selection set is carved out of the
  official `train` split. This preserves the behaviour of every existing dataset.
  Applies to: DermaMNIST, BloodMNIST, RetinaMNIST, **and OCTMNIST** — OCTMNIST's
  test split is only 1,000 examples, too few to evaluate prior adaptation on its
  own, so it is treated like the already-incorporated datasets.

- **`val_role="train"`.** The official `test` split is used *alone* for
  adaptation/evaluation (it is large enough), and the official `val` split
  becomes the **model-selection set** for training/calibration. Applies to the
  three datasets with large test splits: **PathMNIST, TissueMNIST, OrganAMNIST**.

So of the four new datasets, three get `val_role="train"` and OCTMNIST keeps the
default `val_role="test"`.

## 3. `run_base_predictor_exp.py` branch on `spec.val_role`

Today the script *always* carves a stratified model-selection split out of the
training subset (`train_test_split`, `--val-fraction 0.2`) **and** merges on the
presence of a `val` split (`if "val" in ds.splits`). Replace that with an
explicit branch on `spec.val_role`:

- **`val_role == "test"`** — unchanged from current behaviour:
  - model-selection set = stratified carve of `train` (respecting
    `--val-fraction`);
  - fit part = the remainder of `train`;
  - test = `val + test` concatenated when a `val` split exists, else `test`.

- **`val_role == "train"`**:
  - fit part = the **entire** official `train` split (no internal carve; the
    `--val-fraction` argument is unused for these datasets);
  - model-selection set = the official `val` split (used for best-epoch
    selection and for fitting the calibration temperature/bias);
  - test = the official `test` split **only** (no merge).

The training prior `p_tr(y)` is estimated from the fit part exactly as now.

## 4. Required per-dataset metadata

`class_names` is mandatory (`num_classes` derives from its length; the report and
`bincount` use it). `confusable_pair` is needed only if these datasets feed
`run_real_reject_option_exp.py` / `target_prior_search.py`; the values below are
proposals — validate them (or let `--auto-target-prior` pick) before relying on
them. Add an `ARCH_DEFAULTS` entry of `resnet18-28` for each, matching the
existing MedMNIST convention. Class names are transcribed from the MedMNIST v2
`INFO` tables.

**PathMNIST** (`resnet18-28`, RGB, `tags=["rgb", "medical"]`):
```
["adipose", "background", "debris", "lymphocytes", "mucus",
 "smooth muscle", "normal colon mucosa", "cancer-associated stroma",
 "colorectal adenocarcinoma epithelium"]
```
proposed confusable pair: `("cancer-associated stroma", "smooth muscle")`.

**OCTMNIST** (`resnet18-28`, grayscale, `tags=["grayscale", "medical"]`):
```
["choroidal neovascularization", "diabetic macular edema", "drusen", "normal"]
```
proposed confusable pair: `("drusen", "normal")`.

**TissueMNIST** (`resnet18-28`, grayscale, `tags=["grayscale", "medical"]`):
```
["Collecting Duct, Connecting Tubule", "Distal Convoluted Tubule",
 "Glomerular endothelial cells", "Interstitial endothelial cells",
 "Leukocytes", "Podocytes", "Proximal Tubule Segments",
 "Thick Ascending Limb"]
```
proposed confusable pair:
`("Glomerular endothelial cells", "Interstitial endothelial cells")`.

**OrganAMNIST** (`resnet18-28`, grayscale, `tags=["grayscale", "medical"]`):
```
["bladder", "femur-left", "femur-right", "heart", "kidney-left",
 "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
```
proposed confusable pair: `("kidney-left", "kidney-right")` (mirror-image organs
are near-identical at 28×28 — a natural weakly-identifiable pair).

## 5. Notes / out of scope

- `analyze_datasets.py` and `download_datasets.py` need **no** split-policy
  change: download only fetches the raw `.npz`, and analyze reports the three
  splits separately regardless of `val_role`. They pick up the four new keys
  automatically from `DATASETS`.
- The `medmnist` loader already handles grayscale `(N,28,28)` and RGB
  `(N,28,28,3)`, and `to_tensor` in `run_base_predictor_exp.py` adds the channel
  axis for grayscale — no loader changes required.
- TissueMNIST is large (~236k images total); it fits in RAM as uint8 but the
  from-scratch ResNet training run is noticeably longer than the existing
  MedMNIST datasets. Consider `--epochs` / batching expectations accordingly.
