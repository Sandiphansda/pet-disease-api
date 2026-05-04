import numpy as np
import tensorflow as tf
import cv2, json, sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  Load assets once at startup
# ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent / "models"

print(f"[inference] Python  : {sys.version}", flush=True)
print(f"[inference] TF      : {tf.__version__}", flush=True)
print(f"[inference] Model   : {BASE / 'pet_disease_v4.tflite'}", flush=True)

with open(BASE / "labels.json") as f:
    LABELS = json.load(f)

with open(BASE / "scaler.json") as f:
    SCALER = json.load(f)

MEAN  = np.array(SCALER["mean"],  dtype=np.float32)
SCALE = np.array(SCALER["scale"], dtype=np.float32)
print(f"[inference] Scaler mean={MEAN.tolist()}  scale={SCALE.tolist()}", flush=True)
print(f"[inference] Labels ({len(LABELS)}): {list(LABELS.values())}", flush=True)

# ─────────────────────────────────────────────────────────────────
#  Load TFLite model
# ─────────────────────────────────────────────────────────────────
INTERP = tf.lite.Interpreter(model_path=str(BASE / "pet_disease_v4.tflite"))
INTERP.allocate_tensors()

IN_DET  = INTERP.get_input_details()
OUT_DET = INTERP.get_output_details()

# ── Print full tensor details ─────────────────────────────────────
print("\n[inference] === INPUT TENSORS ===", flush=True)
for d in IN_DET:
    print(f"  index={d['index']}  name={d['name']}  shape={d['shape']}  dtype={d['dtype']}", flush=True)
print("[inference] === OUTPUT TENSORS ===", flush=True)
for d in OUT_DET:
    print(f"  index={d['index']}  name={d['name']}  shape={d['shape']}  dtype={d['dtype']}", flush=True)

# ─────────────────────────────────────────────────────────────────
#  Resolve image vs tabular input index by SHAPE, not name
#  image   → 4-dim (1, 224, 224, 3)
#  tabular → 2-dim (1, 27)
# ─────────────────────────────────────────────────────────────────
_img_idx = None
_tab_idx = None

for d in IN_DET:
    if len(d["shape"]) == 4:
        _img_idx = d["index"]
    elif len(d["shape"]) == 2:
        _tab_idx = d["index"]

# Hard positional fallback
if _img_idx is None:
    _img_idx = IN_DET[0]["index"]
if _tab_idx is None and len(IN_DET) > 1:
    _tab_idx = IN_DET[1]["index"]

print(f"[inference] Image  tensor index : {_img_idx}", flush=True)
print(f"[inference] Tabular tensor index: {_tab_idx}", flush=True)

# ─────────────────────────────────────────────────────────────────
#  Disease class index lists per pet
# ─────────────────────────────────────────────────────────────────
DOG_IDX    = [int(i) for i, n in LABELS.items() if "Dog"    in n]
CAT_IDX    = [int(i) for i, n in LABELS.items() if "Cat"    in n or "Feline" in n]
RABBIT_IDX = [int(i) for i, n in LABELS.items() if "Rabbit" in n]
print(f"[inference] Dog idx   : {DOG_IDX}",    flush=True)
print(f"[inference] Cat idx   : {CAT_IDX}",    flush=True)
print(f"[inference] Rabbit idx: {RABBIT_IDX}", flush=True)

SYMPTOM_COLS = [
    "fever", "lethargy", "appetite_loss", "sneezing",
    "nasal_discharge", "lameness", "swallowing", "vomiting",
    "diarrhea", "coughing", "labored_breathing", "skin_lesions",
    "eye_discharge", "dehydration", "loss_of_appetite",
    "reduced_appetite", "swelling", "swollen_joints",
    "swollen_legs", "weight_loss",
]

# ─────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    if not img_bytes:
        print("[inference] No image → zero tensor", flush=True)
        return np.zeros((1, 224, 224, 3), dtype=np.float32)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[inference] Decode failed → zero tensor", flush=True)
        return np.zeros((1, 224, 224, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img[np.newaxis, ...]


def preprocess_tabular(pet_type, age, weight, body_temp, heart_rate, symptoms):
    pet = [
        1.0 if pet_type == "dog"    else 0.0,
        1.0 if pet_type == "cat"    else 0.0,
        1.0 if pet_type == "rabbit" else 0.0,
    ]
    num_raw    = np.array([[age, weight, body_temp, heart_rate]], dtype=np.float32)
    num_scaled = ((num_raw - MEAN) / SCALE)[0].tolist()
    sym        = [float(symptoms.get(c, 0)) for c in SYMPTOM_COLS]
    vec        = pet + num_scaled + sym   # 3 + 4 + 20 = 27 features
    tab        = np.array(vec, dtype=np.float32)[np.newaxis, ...]
    print(f"[inference] Tabular features: {vec}", flush=True)
    return tab


def apply_gate(probs: np.ndarray, pet_type: str) -> np.ndarray:
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


# ─────────────────────────────────────────────────────────────────
#  Predict
# ─────────────────────────────────────────────────────────────────
def predict(img_bytes, pet_type, age, weight,
            body_temp, heart_rate, symptoms, top_k=3):
    print(f"\n[inference] PREDICT  pet={pet_type} age={age} wt={weight} "
          f"temp={body_temp} hr={heart_rate} sym={symptoms}", flush=True)

    img = preprocess_image(img_bytes)
    tab = preprocess_tabular(pet_type, age, weight, body_temp, heart_rate, symptoms)

    # ── Set tensors by pre-resolved index (key fix) ───────────────
    INTERP.set_tensor(_img_idx, img)
    if _tab_idx is not None:
        INTERP.set_tensor(_tab_idx, tab)

    INTERP.invoke()

    raw = INTERP.get_tensor(OUT_DET[0]["index"])[0].copy()
    print(f"[inference] Raw output (all 32): {raw.tolist()}", flush=True)

    probs   = apply_gate(raw, pet_type)
    k       = min(top_k, int((probs > 0).sum()), len(probs))
    top_idx = np.argsort(probs)[::-1][:k]

    results = [
        {
            "rank":       i + 1,
            "disease":    LABELS[str(idx)],
            "confidence": round(float(probs[idx]) * 100, 2),
        }
        for i, idx in enumerate(top_idx)
    ]
    print(f"[inference] Top-{k} results: {results}", flush=True)
    return results
