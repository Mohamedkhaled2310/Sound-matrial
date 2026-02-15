import os
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
MODEL_PATH = os.environ.get("MODEL_PATH", "./material_sound_model_v2.h5")

app = FastAPI()

yamnet_model = None
model = None


def convert_to_wav_16bit(input_path: str) -> str:
    sound = AudioSegment.from_file(input_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(
        temp_wav.name, format="wav"
    )
    return temp_wav.name


@app.on_event("startup")
def startup():
    global yamnet_model, model
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    model = tf.keras.models.load_model(MODEL_PATH)


def load_audio(file_path: str) -> tf.Tensor:
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # (اختياري) pad لو الصوت قصير جدًا (مثلاً أقل من 1 ثانية)
    min_len = 16000
    audio_len = tf.shape(audio)[0]
    audio = tf.cond(
        audio_len < min_len,
        lambda: tf.pad(audio, [[0, min_len - audio_len]]),
        lambda: audio
    )

    return audio


def extract_embedding(audio: tf.Tensor) -> tf.Tensor:
    _, embeddings, _ = yamnet_model(audio)
    return tf.reduce_mean(embeddings, axis=0)


def predict_material(file_path: str):
    audio = load_audio(file_path)
    embedding = extract_embedding(audio)
    embedding = tf.expand_dims(embedding, axis=0)

    preds = model.predict(embedding, verbose=0)[0]
    pred_class = CLASS_NAMES[int(np.argmax(preds))]
    pred_probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return pred_class, pred_probs


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict-audio/")
async def predict_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    safe_wav = None
    try:
        safe_wav = convert_to_wav_16bit(tmp_path)
        pred_class, pred_probs = predict_material(safe_wav)
        return JSONResponse({"predicted_class": pred_class, "probabilities": pred_probs})
    finally:
        for p in [tmp_path, safe_wav]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
