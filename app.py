#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit Inference App for Sanskrit ASR
- Supports WAV upload
- Supports microphone recording (Windows-safe via streamlit-webrtc)
- Uses wav2vec2 (models/wav2vec_hindi)
- Uses finetuned Transformer ASR (sanskrit_asr_model/finetuned_sanskrit_asr.pt)
"""

import os
import tempfile
import queue
import numpy as np
import torch
import soundfile as sf
import librosa
import streamlit as st
import av

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------------------------------------------------------
# PROJECT ROOT (RELATIVE PATHS)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

WAV2VEC_PATH = os.path.join(PROJECT_ROOT, "models", "wav2vec_hindi")
ASR_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "sanskrit_asr_model", "finetuned_sanskrit_asr.pt"
)

# ---------------------------------------------------------
# IMPORT TRAINING CONFIG (MUST MATCH TRAINING EXACTLY)
# ---------------------------------------------------------
from train_transformer_asr import (
    TransformerASR,
    vectorizer,
    max_target_len,
    feature_hidden_size,
    num_hid,
    num_head,
    num_feed_forward,
    source_maxlen,
    num_layers_enc,
    num_layers_dec,
    drop_out_enc,
    drop_out_dec,
    drop_out_cross,
    target_start_token_idx,
    target_end_token_idx,
)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
SAMPLE_RATE = 16000
MAX_DURATION = 12  # seconds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# AUDIO HELPERS
# ---------------------------------------------------------
def fix_audio_length(audio, expected_len):
    if len(audio) < expected_len:
        return np.pad(audio, (0, expected_len - len(audio)))
    return audio[:expected_len]


def decode_idxs(idxs, idx_to_token):
    tokens = []
    for idx in idxs:
        idx = int(idx)
        if idx == 0:
            continue
        tok = idx_to_token(idx)
        tokens.append(tok)
        if idx == target_end_token_idx:
            break
    return "".join(tokens).replace("-", "")

# ---------------------------------------------------------
# LOAD MODELS (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_PATH)
    wav2vec = Wav2Vec2Model.from_pretrained(WAV2VEC_PATH).to(DEVICE)
    wav2vec.eval()

    num_classes = vectorizer.get_vocabulary_size()
    model = TransformerASR(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        source_maxlen=source_maxlen,
        target_maxlen=max_target_len,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=num_classes,
        feature_dim=feature_hidden_size,
        drop_out_enc=drop_out_enc,
        drop_out_dec=drop_out_dec,
        drop_out_cross=drop_out_cross,
    ).to(DEVICE)

    state = torch.load(ASR_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return extractor, wav2vec, model

# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------
@torch.no_grad()
def transcribe(wav_path):
    extractor, wav2vec, model = load_models()

    audio, sr = sf.read(wav_path)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, sr, SAMPLE_RATE)

    expected_len = SAMPLE_RATE * MAX_DURATION
    audio = fix_audio_length(audio, expected_len)

    ex = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    ex = {k: v.to(DEVICE) for k, v in ex.items()}

    out = wav2vec(**ex)
    features = out.last_hidden_state.squeeze(0)

    mask = wav2vec._get_feature_vector_attention_mask(
        features.size(0), ex["attention_mask"]
    ).squeeze(0).bool()

    preds = model.generate(
        features.unsqueeze(0),
        target_start_token_idx,
        target_end_token_idx,
        source_mask=mask.unsqueeze(0),
    )

    idx_to_token = vectorizer.idx_to_token()
    return decode_idxs(preds[0].cpu(), idx_to_token)

# ---------------------------------------------------------
# STREAMLIT-WEBRTC AUDIO PROCESSOR
# ---------------------------------------------------------
class AudioProcessor:
    def __init__(self):
        self.audio_buffer = queue.Queue()

    def recv(self, frame: av.AudioFrame):
        pcm = frame.to_ndarray()
        self.audio_buffer.put(pcm)
        return frame

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Sanskrit ASR", layout="centered")
st.title("🕉️ Sanskrit Automatic Speech Recognition")

st.markdown(
    """
Upload a **Sanskrit WAV file** or **record from your microphone**.  
The system uses **wav2vec2 + Transformer ASR** trained on **Vāksañcayaḥ**.
"""
)

# ---------------------------------------------------------
# WAV UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader("📂 Upload WAV file", type=["wav"])

wav_path = None

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded.read())
        wav_path = f.name

# ---------------------------------------------------------
# MICROPHONE (WINDOWS-SAFE)
# ---------------------------------------------------------
st.markdown("### 🎙️ Record from Microphone")

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if ctx.audio_processor:
    if st.button("🛑 Stop & Transcribe"):
        audio_frames = []
        while not ctx.audio_processor.audio_buffer.empty():
            audio_frames.append(ctx.audio_processor.audio_buffer.get())

        if audio_frames:
            audio = np.concatenate(audio_frames, axis=1).T.astype(np.float32)
            audio /= np.max(np.abs(audio)) + 1e-9

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, audio, SAMPLE_RATE)
                wav_path = f.name

# ---------------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------------
if wav_path:
    with st.spinner("Transcribing..."):
        transcript = transcribe(wav_path)

    st.success("✅ Transcription Complete")
    st.text_area("📜 Devanagari Transcript", transcript, height=180)
