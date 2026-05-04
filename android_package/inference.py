import numpy as np
import tensorflow as tf
import cv2, json
from pathlib import Path

# ── Load once at startup ──────────────────────────────────────
BASE = Path(__file__).parent.parent / "models"

with open(BASE / "labels.json")  as f: LABELS  = json.load(f)
with open(BASE / "scaler.json")  as f: SCALER  = json.load(f)

MEAN  = np.array(SCALER["mean"],  dtype=np.float32)
SCALE = np.array(SCALER["scale"], dtype=np.float32)

INTERP = tf.lite.Interpreter(
    model_path=str(BASE / "pet_disease_v4.tflite")
)
INTERP.allocate_tensors()
IN_DET  = INTERP.get_input_details()
OUT_DET = INTERP.get_output_details()

# ── Debug: print tensor names so you know the real names ──────
print("=== TFLite Input Tensors ===")
for i, d in enumerate(IN_DET):
    print(f"  [{i}] name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")
print("=== TFLite Output Tensors ===")
for i, d in enumerate(OUT_DET):
    print(f"  [{i}] name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")

SYMPTOM_COLS = [
    "fever","lethargy","appetite_loss","sneezing",
    "nasal_discharge","lameness","swallowing","vomiting",
    "diarrhea","coughing","labored_breathing","skin_lesions",
    "eye_discharge","dehydration","loss_of_appetite",
    "reduced_appetite","swelling","swollen_joints",
    "swollen_legs","weight_loss",
]

# disease index groups per pet  (LABELS keys are strings "0".."31")
DOG_IDX    = [int(i) for i,n in LABELS.items() if "Dog"    in n]
CAT_IDX    = [int(i) for i,n in LABELS.items() if "Cat"    in n or "Feline" in n]
RABBIT_IDX = [int(i) for i,n in LABELS.items() if "Rabbit" in n]


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Return (1, 224, 224, 3) float32 normalised image, or zeros if no image."""
    if not img_bytes:
        return np.zeros((1, 224, 224, 3), dtype=np.float32)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((1, 224, 224, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5          # MobileNetV2 normalisation
    return img[np.newaxis, ...]       # (1, 224, 224, 3)


def preprocess_tabular(pet_type, age, weight,
                        body_temp, heart_rate,
                        symptoms: dict) -> np.ndarray:
    """Return (1, 27) float32 tabular feature vector."""
    pet = [
        1.0 if pet_type == "dog"    else 0.0,
        1.0 if pet_type == "cat"    else 0.0,
        1.0 if pet_type == "rabbit" else 0.0,
    ]
    num_raw    = np.array([[age, weight, body_temp, heart_rate]], dtype=np.float32)
    num_scaled = ((num_raw - MEAN) / SCALE)[0].tolist()
    sym        = [float(symptoms.get(c, 0)) for c in SYMPTOM_COLS]
    tab        = np.array(pet + num_scaled + sym, dtype=np.float32)[np.newaxis, ...]
    return tab   # (1, 27)


def apply_gate(probs: np.ndarray, pet_type: str) -> np.ndarray:
    """Zero out disease classes that don't belong to the selected pet type."""
    masked = probs.copy()
    if pet_type == "dog":
        for i in CAT_IDX + RABBIT_IDX:
            masked[i] = 0.0
    elif pet_type == "cat":
        for i in DOG_IDX + RABBIT_IDX:
            masked[i] = 0.0
    elif pet_type == "rabbit":
        for i in DOG_IDX + CAT_IDX:
            masked[i] = 0.0
    total = masked.sum()
    if total > 0:
        masked /= total
    return masked


def _find_input_index(shape_len: int) -> int:
    """
    Return the IN_DET index whose tensor has `shape_len` dimensions.
    Image input has 4 dims (1,224,224,3); tabular has 2 dims (1,27).
    Falls back to positional order if ambiguous.
    """
    for i, d in enumerate(IN_DET):
        if len(d["shape"]) == shape_len:
            return i
    return 0   # fallback


def predict(img_bytes, pet_type, age, weight,
            body_temp, heart_rate, symptoms, top_k=3):
    """
    Run inference and return a list of top_k dicts:
      [{"rank": 1, "disease": "...", "confidence": 58.2}, ...]

    FIX: Inputs are assigned by tensor shape, NOT by tensor name.
    The original code checked inp["name"] for "image"/"tabular" which
    fails because TFLite renames tensors during conversion.
    """
    img = preprocess_image(img_bytes)
    tab = preprocess_tabular(pet_type, age, weight,
                              body_temp, heart_rate, symptoms)

    # ── Assign inputs by shape (robust, name-independent) ────────
    img_input_idx = _find_input_index(4)   # 4-dim = image (1,224,224,3)
    tab_input_idx = _find_input_index(2)   # 2-dim = tabular (1,27)

    # Safety: if both resolve to the same index (single-input model),
    # fall back to positional: 0=image, 1=tabular
    if img_input_idx == tab_input_idx:
        img_input_idx = 0
        tab_input_idx = 1 if len(IN_DET) > 1 else 0

    INTERP.set_tensor(IN_DET[img_input_idx]["index"], img)
    if len(IN_DET) > 1:
        INTERP.set_tensor(IN_DET[tab_input_idx]["index"], tab)

    INTERP.invoke()

    probs   = INTERP.get_tensor(OUT_DET[0]["index"])[0].copy()
    probs   = apply_gate(probs, pet_type)

    # Get top-k indices sorted by probability descending
    k       = min(top_k, len(probs))
    top_idx = np.argsort(probs)[::-1][:k]

    return [
        {
            "rank":       i + 1,
            "disease":    LABELS[str(idx)],
            "confidence": round(float(probs[idx]) * 100, 2),
        }
        for i, idx in enumerate(top_idx)
    ]
