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
    # Role of the official ``val`` split in ``base_predictor_training.py``:
    # "test" merges val into the test subset (evaluation only) and carves the
    # model-selection set out of train (default, current behaviour); "train"
    # trains on the whole official train split, uses the official val split as
    # the model-selection set, and evaluates on the official test split alone.
    val_role: str = "test"


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
    "cifar100": DatasetSpec(
        key="cifar100",
        # Same fast.ai PNG image-folder mirror as CIFAR-10.
        # Layout: cifar100/<split>/<fine-class>/*.png (100 fine classes).
        display_name="CIFAR-100",
        kind="imagefolder",
        files=[
            ("https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz", "cifar100.tgz"),
        ],
        archive_dir="cifar100",
        class_names=[
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
            "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
            "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
            "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
            "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
            "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
            "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
            "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
            "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
            "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
            "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
            "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
            "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
            "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
            "willow_tree", "wolf", "woman", "worm",
        ],
        description=(
            "100 fine-grained classes at 32x32, with 20 superclasses giving "
            "naturally confusable within-group pairs (boy/girl, the maple/oak/"
            "palm/pine/willow trees, bicycle/motorcycle) and 98 other classes "
            "as decoys. boy/girl is near-indistinguishable at this resolution "
            "and carries a plausible demographic prevalence-shift story. Use "
            "pretrained frozen features + a logistic head, as for CIFAR-10."
        ),
        confusable_pair=("boy", "girl"),
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
    "tissuemnist": DatasetSpec(
        key="tissuemnist",
        display_name="TissueMNIST (MedMNIST v2)",
        kind="medmnist",
        files=[(_ZENODO + "tissuemnist.npz?download=1", "tissuemnist.npz")],
        class_names=[
            "Collecting Duct, Connecting Tubule",
            "Distal Convoluted Tubule",
            "Glomerular endothelial cells",
            "Interstitial endothelial cells",
            "Leukocytes",
            "Podocytes",
            "Proximal Tubule Segments",
            "Thick Ascending Limb",
        ],
        description=(
            "Human kidney cortex cells, 8 tissue types. The two endothelial cell "
            "types (glomerular vs. interstitial) are near-identical at 28x28, a "
            "natural weakly-identifiable pair. Large test split, so the official "
            "val split is used for training."
        ),
        confusable_pair=(
            "Glomerular endothelial cells",
            "Interstitial endothelial cells",
        ),
        tags=["grayscale", "medical"],
        val_role="train",
    ),
    "organamnist": DatasetSpec(
        key="organamnist",
        display_name="OrganAMNIST (MedMNIST v2)",
        kind="medmnist",
        files=[(_ZENODO + "organamnist.npz?download=1", "organamnist.npz")],
        class_names=[
            "bladder",
            "femur-left",
            "femur-right",
            "heart",
            "kidney-left",
            "kidney-right",
            "liver",
            "lung-left",
            "lung-right",
            "pancreas",
            "spleen",
        ],
        description=(
            "Abdominal CT organs (axial view), 11 classes. Mirror-image organs "
            "(left vs. right kidney) are near-identical at 28x28 -- a natural "
            "weakly-identifiable pair -- while the other organs act as decoys. "
            "Large test split, so the official val split is used for training."
        ),
        confusable_pair=("kidney-left", "kidney-right"),
        tags=["grayscale", "medical"],
        val_role="train",
    ),
    "organsmnist": DatasetSpec(
        key="organsmnist",
        display_name="OrganSMNIST (MedMNIST v2)",
        kind="medmnist",
        files=[(_ZENODO + "organsmnist.npz?download=1", "organsmnist.npz")],
        class_names=[
            "bladder",
            "femur-left",
            "femur-right",
            "heart",
            "kidney-left",
            "kidney-right",
            "liver",
            "lung-left",
            "lung-right",
            "pancreas",
            "spleen",
        ],
        description=(
            "Abdominal CT organs (sagittal view), 11 classes. Same organ labels "
            "as OrganAMNIST but sliced sagittally; mirror-image organs (left vs. "
            "right kidney) stay a natural weakly-identifiable pair while the "
            "other organs act as decoys. The official train/val/test split is "
            "kept intact -- the val split is the model-selection set and "
            "evaluation uses the test split alone (val is not merged into test)."
        ),
        confusable_pair=("kidney-left", "kidney-right"),
        tags=["grayscale", "medical"],
        val_role="train",
    ),
}
