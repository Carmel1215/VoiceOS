import sounddevice as sd
from faster_whisper import WhisperModel

SR = 16000
DURATION = 3.0
MODEL_SIZE = 'medium'  # tiny/base/small/medium/large-v3

def load_model():
    return WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')

def record_audio(duration=DURATION, sr=SR):
    print('listening...')
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()

def transcribe(model, audio):
    segments, info = model.transcribe(audio, language="ko", vad_filter=True)
    return ''.join(seg.text for seg in segments).strip()