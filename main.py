# main.py - PetDiseaseV4 FastAPI server
import os, json, pickle, io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import uvicorn

app = FastAPI(
    title       = "PetDiseaseV4 API",
    description = "Predict disease in Dog, Cat, Rabbit from image + symptoms",
    version     = "4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load model and metadata on startup ───────────────────────
BASE     = os.path.dirname(__file__)
TFLITE   = os.path.join(BASE, "tflite",    "pet_disease_v4.tflite")
LABELS   = os.path.join(BASE, "tflite",    "feature_meta.json")
META     = os.path.join(BASE, "metadata",  "scaler.pkl")
LE_PATH  = os.path.join(BASE, "metadata",  "label_encoder.pkl")

# Load TFLite
interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_details  = interp.get_input_details()
out_details = interp.get_output_details()

# Load scaler and label encoder
with open(META,    'rb') as f: scaler = pickle.load(f)
with open(LE_PATH, 'rb') as f: le     = pickle.load(f)

label_map = {i: name for i, name in enumerate(le.classes_)}

print(f"Model loaded: {len(label_map)} classes")
print(f"Input details: {[(d['name'], d['shape']) for d in in_details]}")

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE         = (224, 224)
PET_ONE_HOT      = ['animal_dog', 'animal_cat', 'animal_rabbit']
NUMERIC_COLS     = ['age', 'weight', 'body_temperature', 'heart_rate']
CSV_SYMPTOM_COLS = [
    'fever', 'lethargy', 'appetite_loss', 'sneezing', 'nasal_discharge',
    'lameness', 'swallowing', 'vomiting', 'diarrhea', 'coughing',
    'labored_breathing', 'skin_lesions', 'eye_discharge', 'dehydration',
    'loss_of_appetite', 'reduced_appetite', 'swelling', 'swollen_joints',
    'swollen_legs', 'weight_loss',
]

DOG_IDX    = [i for i, n in label_map.items() if 'Dog'    in n]
CAT_IDX    = [i for i, n in label_map.items() if 'Cat'    in n or 'Feline' in n]
RABBIT_IDX = [i for i, n in label_map.items() if 'Rabbit' in n]

# ── Helpers ───────────────────────────────────────────────────
def preprocess_image(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img[np.newaxis, ...]

def build_tabular(pet_type, age, weight, body_temp,
                  heart_rate, symptoms: dict):
    pet_vec = [
        1 if pet_type == 'dog'    else 0,
        1 if pet_type == 'cat'    else 0,
        1 if pet_type == 'rabbit' else 0,
    ]
    num_raw    = np.array([[age, weight, body_temp, heart_rate]],
                           dtype=np.float32)
    num_scaled = scaler.transform(num_raw)[0].tolist()
    sym_vec    = [float(symptoms.get(c, 0)) for c in CSV_SYMPTOM_COLS]
    tab = np.array(pet_vec + num_scaled + sym_vec,
                   dtype=np.float32)[np.newaxis, ...]
    return tab

def apply_gate(probs, pet_type):
    masked = probs.copy()
    if pet_type == 'dog':
        for i in CAT_IDX + RABBIT_IDX:
            masked[i] = 0.0
    elif pet_type == 'cat':
        for i in DOG_IDX + RABBIT_IDX:
            masked[i] = 0.0
    elif pet_type == 'rabbit':
        for i in DOG_IDX + CAT_IDX:
            masked[i] = 0.0
    total = masked.sum()
    if total > 0:
        masked /= total
    return masked

def run_inference(img_arr, tab_arr):
    for inp in in_details:
        if 'image' in inp['name'].lower():
            interp.set_tensor(inp['index'], img_arr)
        else:
            interp.set_tensor(inp['index'], tab_arr)
    interp.invoke()
    return interp.get_tensor(out_details[0]['index'])[0]

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "running",
        "version": "4.0",
        "classes": len(label_map),
        "pets":    ["dog", "cat", "rabbit"],
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET  /health",
            "labels":  "GET  /labels",
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "PetDiseaseV4"}

@app.get("/labels")
def get_labels():
    return {"labels": label_map, "total": len(label_map)}

@app.post("/predict")
async def predict(
    pet_type     : str           = Form(...),
    age          : float         = Form(...),
    weight       : float         = Form(...),
    body_temp    : float         = Form(...),
    heart_rate   : float         = Form(...),
    fever                : int   = Form(0),
    lethargy             : int   = Form(0),
    appetite_loss        : int   = Form(0),
    sneezing             : int   = Form(0),
    nasal_discharge      : int   = Form(0),
    lameness             : int   = Form(0),
    swallowing           : int   = Form(0),
    vomiting             : int   = Form(0),
    diarrhea             : int   = Form(0),
    coughing             : int   = Form(0),
    labored_breathing    : int   = Form(0),
    skin_lesions         : int   = Form(0),
    eye_discharge        : int   = Form(0),
    dehydration          : int   = Form(0),
    loss_of_appetite     : int   = Form(0),
    reduced_appetite     : int   = Form(0),
    swelling             : int   = Form(0),
    swollen_joints       : int   = Form(0),
    swollen_legs         : int   = Form(0),
    weight_loss          : int   = Form(0),
    image : Optional[UploadFile] = File(None),
):
    # validate pet type
    if pet_type not in ['dog', 'cat', 'rabbit']:
        raise HTTPException(400, "pet_type must be dog, cat or rabbit")

    # image
    if image is not None:
        img_bytes = await image.read()
        img_arr   = preprocess_image(img_bytes)
    else:
        img_arr   = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)

    # tabular
    symptoms = {
        'fever': fever, 'lethargy': lethargy,
        'appetite_loss': appetite_loss, 'sneezing': sneezing,
        'nasal_discharge': nasal_discharge, 'lameness': lameness,
        'swallowing': swallowing, 'vomiting': vomiting,
        'diarrhea': diarrhea, 'coughing': coughing,
        'labored_breathing': labored_breathing,
        'skin_lesions': skin_lesions, 'eye_discharge': eye_discharge,
        'dehydration': dehydration, 'loss_of_appetite': loss_of_appetite,
        'reduced_appetite': reduced_appetite, 'swelling': swelling,
        'swollen_joints': swollen_joints, 'swollen_legs': swollen_legs,
        'weight_loss': weight_loss,
    }
    tab_arr = build_tabular(pet_type, age, weight,
                             body_temp, heart_rate, symptoms)

    # inference
    probs  = run_inference(img_arr, tab_arr)
    probs  = apply_gate(probs, pet_type)

    top3_idx = np.argsort(probs)[::-1][:3]
    top3     = [
        {
            "disease":    label_map[int(i)],
            "confidence": round(float(probs[i]) * 100, 2)
        }
        for i in top3_idx
    ]

    return {
        "status":     "success",
        "pet_type":   pet_type,
        "prediction": top3[0]["disease"],
        "confidence": top3[0]["confidence"],
        "top3":       top3,
        "mode":       "image+symptoms" if image else "symptoms_only"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)