import os
import torch


def _get_env(name, default, cast):
    value = os.environ.get(name)
    if value is None:
        return default
    return cast(value)


def _detect_runtime_defaults():
    logical_cpu = os.cpu_count() or 4
    has_cuda = torch.cuda.is_available()
    vram_gb = 0.0
    if has_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if has_cuda and vram_gb >= 11.0:
        return {
            "img_size": 640,
            "batch_size": 4,
            "val_max_batches": 16,
            "num_workers": min(8, max(4, logical_cpu // 2)),
            "pin_memory": 1,
            "prefetch_factor": 2,
            "persistent_workers": 1,
        }
    if has_cuda and vram_gb >= 7.0:
        return {
            "img_size": 640,
            "batch_size": 2,
            "val_max_batches": 12,
            "num_workers": min(4, max(2, logical_cpu // 2)),
            "pin_memory": 1,
            "prefetch_factor": 2,
            "persistent_workers": 1,
        }
    return {
        "img_size": 512,
        "batch_size": 1,
        "val_max_batches": 8,
        "num_workers": min(2, max(0, logical_cpu // 4)),
        "pin_memory": 0,
        "prefetch_factor": 2,
        "persistent_workers": 0,
    }


# 경로 설정 (환경변수 우선, 없으면 프로젝트 기준 기본값)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
BASE_DIR = os.environ.get("VISDRONE_BASE_DIR", DATASETS_DIR)
TRAIN_DIR = os.environ.get("VISDRONE_TRAIN_DIR", os.path.join(BASE_DIR, "VisDrone2019-DET-train"))
VAL_DIR = os.environ.get("VISDRONE_VAL_DIR", os.path.join(BASE_DIR, "VisDrone2019-DET-val"))
YOLO_PATH = os.environ.get("YOLO_PATH", os.path.join(PROJECT_DIR, "weights", "detectors", "visdrone_best.pt"))
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", "manual")
RESULTS_CSV = os.environ.get("RESULTS_CSV", os.path.join(OUTPUTS_DIR, "training", "experiment_results.csv"))
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(OUTPUTS_DIR, "training", "generators", f"{EXPERIMENT_ID.lower()}results"),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_RUNTIME_DEFAULTS = _detect_runtime_defaults()
IMG_SIZE = _get_env("IMG_SIZE", _RUNTIME_DEFAULTS["img_size"], int)
BATCH_SIZE = _get_env("BATCH_SIZE", _RUNTIME_DEFAULTS["batch_size"], int)
LEARNING_RATE = _get_env("LEARNING_RATE", 1e-4, float)
EPOCHS = _get_env("EPOCHS", 20, int)
EARLY_STOP_PATIENCE = _get_env("EARLY_STOP_PATIENCE", 4, int)
VAL_MAX_BATCHES = _get_env("VAL_MAX_BATCHES", _RUNTIME_DEFAULTS["val_max_batches"], int)
NUM_WORKERS = _get_env("NUM_WORKERS", _RUNTIME_DEFAULTS["num_workers"], int)
PIN_MEMORY = bool(_get_env("PIN_MEMORY", _RUNTIME_DEFAULTS["pin_memory"], int))
PREFETCH_FACTOR = _get_env("PREFETCH_FACTOR", _RUNTIME_DEFAULTS["prefetch_factor"], int)
PERSISTENT_WORKERS = bool(_get_env("PERSISTENT_WORKERS", _RUNTIME_DEFAULTS["persistent_workers"], int))

# Attack budgets
EPSILON_V = _get_env("EPSILON_V", 16 / 255.0, float)  # Vanish noise budget
EPSILON_F = _get_env("EPSILON_F", 4 / 255.0, float)   # Fabricate noise budget

# Fabricate composite loss weights
HINGE_CAP = _get_env("HINGE_CAP", 0.5, float)
LAMBDA_L2 = _get_env("LAMBDA_L2", 15.0, float)
GAMMA_TV = _get_env("GAMMA_TV", 1.5, float)
LAMBDA_COL = _get_env("LAMBDA_COL", 3.0, float)

# Vanish stealth regularization weights
VANISH_LAMBDA_L2 = _get_env("VANISH_LAMBDA_L2", 0.0, float)
VANISH_GAMMA_TV = _get_env("VANISH_GAMMA_TV", 0.0, float)
VANISH_LAMBDA_COL = _get_env("VANISH_LAMBDA_COL", 0.0, float)

STRIDES = [8, 16, 32]
