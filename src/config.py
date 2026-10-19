"""Chemins, classes, mapping label -> chiffre, seeds et interrupteurs des leviers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
LABELS_CSV = ROOT / "train_labels.csv"

OUT = ROOT / "outputs"
INDEX_PATH = OUT / "index" / "index.parquet"
OOF_DIR = OUT / "oof"
TEST_PROBS_DIR = OUT / "test_probs"
MODELS_DIR = OUT / "models"
SUB_DIR = OUT / "submissions"
FIG_DIR = OUT / "figures"
EMB_DIR = OUT / "embeddings"
for d in (INDEX_PATH.parent, OOF_DIR, TEST_PROBS_DIR, MODELS_DIR, SUB_DIR, FIG_DIR, EMB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Ordre de l'énoncé -> chiffre soumis (hypothèse validée par la 1re soumission).
CLASSES = ["apple", "google", "whatsapp", "facebook", "samsung", "mozilla", "messenger"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

SEED = 42
N_FOLDS = 5

# Interrupteurs des leviers (utilisés par make_submission.py et ablation.py).
LEVERS = {
    "use_meta": True,            # métadonnées PNG dans le stacking
    "copy_duplicates": True,     # label du jumeau train copié sur le test
    # Un vendeur par concept (affectation hongroise) : désactivé, la simulation OOF
    # donne -0.6 à -2.1 pts (variantes d'un même vendeur fusionnées dans les groupes).
    # La version douce (features kNN dans le stacking) est conservée.
    "concept_constraint": False,
}

# CNN dont les poids par fold sont sauvegardés dans outputs/models (<nom>_f<k>.pt) :
# architecture timm et résolution d'entrée, utilisées par predict.py.
CNN_MODELS = {
    "cnx_t160": ("convnext_tiny.fb_in22k_ft_in1k", 160),
    "cnx_t224_pl": ("convnext_tiny.fb_in22k_ft_in1k", 224),
    "cnx_t160_pl": ("convnext_tiny.fb_in22k_ft_in1k", 160),
}
