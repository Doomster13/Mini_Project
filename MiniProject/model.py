import os
import time
import uuid
import torch
import librosa
import whisper
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from demucs.pretrained import get_model
from demucs.apply import apply_model

# ------------------ CONFIG ------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ STEP 1: Speech Isolation ------------------
def isolate_audio(input_file):
    model = get_model("htdemucs")
    model.eval()

    wav, sr = torchaudio.load(input_file)

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    sources = apply_model(model, wav[None], device='cpu')[0]
    vocals = sources[3]

    # Unique filename (UUID + timestamp for safety)
    unique_id = uuid.uuid4().hex
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"clean_voice_{timestamp}_{unique_id}.wav")

    torchaudio.save(output_path, vocals, sr)

    return output_path

# ------------------ STEP 2: Speech to Text ------------------
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# ------------------ STEP 3: Emotion Detection ------------------
def detect_emotion(audio_file):
    import transformers
    transformers.utils.import_utils._torch_available = True

    model_name = "superb/wav2vec2-base-superb-er"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    audio, sr = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[predicted_id]

# ------------------ MAIN PIPELINE ------------------
def run_pipeline(input_file):
    clean_audio = isolate_audio(input_file)
    text = transcribe_audio(clean_audio)
    emotion = detect_emotion(clean_audio)

    return {
        "text": text,
        "emotion": emotion,
        "audio": clean_audio
    }

# ------------------ RUN ------------------
if __name__ == "__main__":
    result = run_pipeline("input.wav")
    print(result)