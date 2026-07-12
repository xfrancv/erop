"""Static metadata for each candidate dataset: download URLs and class names.

Class-name lists for the MedMNIST datasets are transcribed from the official
MedMNIST v2 ``INFO`` tables (the ``.npz`` archives carry only integer labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset lives and what its labels mean.

    ``kind`` selects the loader in :mod:`data_tools.loaders`:
    ``"idx"`` (Fashion-MNIST IDX files), ``"imagefolder"`` (a tarball of
    ``<split>/<class>/*.png``, e.g. CIFAR-10) or ``"medmnist"`` (a single
    ``.npz`` archive).
    """

    key: str
    display_name: str
    kind: str
    # (url, local filename) pairs to fetch into data/<key>/.
    files: list[tuple[str, str]]
    class_names: list[str]
    description: str
    # For "cifar": the tar member directory to read after extraction.
    archive_dir: str = ""
    confusable_pair: tuple[str, str] | None = None
    tags: list[str] = field(default_factory=list)


_FASHION_BASE = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    "master/data/fashion/"
)

_ZENODO = "https://zenodo.org/records/10519652/files/"


DATASETS: dict[str, DatasetSpec] = {
    "fashion_mnist": DatasetSpec(
        key="fashion_mnist",
        display_name="Fashion-MNIST",
        kind="idx",
        files=[
            (_FASHION_BASE + "train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
            (_FASHION_BASE + "train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
            (_FASHION_BASE + "t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
            (_FASHION_BASE + "t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
        ],
        class_names=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        description=(
            "Zalando article images, a drop-in MNIST replacement. The proposal's "
            "cheap first step: the {t-shirt, pullover, shirt, coat} group is "
            "heavily confusable and runs with the existing logistic-regression "
            "base model."
        ),
        confusable_pair=("Shirt", "T-shirt/top"),
        tags=["grayscale", "cheap"],
    ),
    "cifar10": DatasetSpec(
        key="cifar10",
        # PNG image-folder mirror (fast.ai); the canonical Toronto tarball is
        # served too slowly to be usable. Layout: cifar10/<split>/<class>/*.png.
        display_name="CIFAR-10",
        kind="imagefolder",
        files=[
            ("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz", "cifar10.tgz"),
        ],
        archive_dir="cifar10",
        class_names=[
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        description=(
            "The recognizable vision benchmark. Cat/dog is the classic weakly "
            "identifiable pair under a moderate base model; deer/horse/bird "
            "supply naturally hard, regret-free decoys. Intended to be used with "
            "pretrained frozen features + a logistic head."
        ),
        confusable_pair=("cat", "dog"),
        tags=["rgb"],
    ),
    "dermamnist": DatasetSpec(
        key="dermamnist",
        display_name="DermaMNIST (MedMNIST v2)",
        kind="medmnist",
        files=[(_ZENODO + "dermamnist.npz?download=1", "dermamnist.npz")],
        class_names=[
            "actinic keratoses / intraepithelial carcinoma",
            "basal cell carcinoma",
            "benign keratosis-like lesions",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions",
        ],
        description=(
            "Dermatoscopic images (HAM10000). Melanoma vs. benign nevus is a "
            "notoriously near-identical, weakly identifiable pair whose "
            "prevalence varies enormously between screening and referral "
            "populations — the medical label-shift story to lead with."
        ),
        confusable_pair=("melanoma", "melanocytic nevi"),
        tags=["rgb", "medical"],
    ),
    "bloodmnist": DatasetSpec(
        key="bloodmnist",
        display_name="BloodMNIST (MedMNIST v2)",
        kind="medmnist",
        files=[(_ZENODO + "bloodmnist.npz?download=1", "bloodmnist.npz")],
        class_names=[
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature granulocytes",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet",
        ],
        description=(
            "Peripheral blood cell microscopy images. Cell-type prevalence "
            "shifts dramatically with infection (e.g. neutrophils), with the "
            "other cell types acting as aleatoric decoys."
        ),
        confusable_pair=("neutrophil", "immature granulocytes"),
        tags=["rgb", "medical"],
    ),
}
