import os, subprocess, zipfile, shutil, uuid
from flask import Flask, request, send_file, render_template_string
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt

app = Flask(__name__)
UPLOAD = "uploads"
OUT = "outputs"
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

HTML = open("templates/index.html").read() if os.path.exists("templates/index.html") else "<h1>Sube tu cancion</h1><form method=post enctype=multipart/form-data action=/separate><input type=file name=file><button>Separar</button></form>"

@app.route("/")
def index():
    return render_template_string(HTML)

def bandpass(y, sr, low, high):
    sos = butter(4, [low, high], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y)

def lowpass(y, sr, cutoff):
    sos = butter(4, cutoff, btype='lowpass', fs=sr, output='sos')
    return sosfilt(sos, y)

def highpass(y, sr, cutoff):
    sos = butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, y)

@app.route("/separate", methods=["POST"])
def separate():
    f = request.files.get("file")
    if not f:
        return "No subiste archivo", 400
    uid = str(uuid.uuid4())[:8]
    in_path = os.path.join(UPLOAD, uid + "_" + f.filename)
    f.save(in_path)
    # 1. Demucs 6 stems
    subprocess.run(["python", "-m", "demucs", "-n", "htdemucs_6s", "-o", OUT, in_path], check=True)
    base = os.path.splitext(os.path.basename(in_path))[0]
    demucs_dir = os.path.join(OUT, "htdemucs_6s", base)
    # Demucs files: drums.wav, bass.wav, other.wav, vocals.wav, piano.wav, guitar.wav
    def load(name):
        p = os.path.join(demucs_dir, name + ".wav")
        y, sr = librosa.load(p, sr=44100, mono=False)
        return y, sr
    stems = {}
    try:
        drums, sr = load("drums")
        bass, _ = load("bass")
        other, _ = load("other")
        vocals, _ = load("vocals")
        piano, _ = load("piano")
        guitar, _ = load("guitar")
    except Exception as e:
        return f"Error Demucs: {e}", 500

    # 2. Derivar 4 stems de bateria desde drums
    # drums shape: (2, samples) o (samples,)
    if drums.ndim == 1:
        drums = np.stack([drums, drums])
    # Procesar por canal
    kick = np.stack([lowpass(drums[ch], sr, 150) for ch in range(drums.shape[0])])
    snare = np.stack([bandpass(drums[ch], sr, 150, 2500) for ch in range(drums.shape[0])])
    hihat = np.stack([highpass(drums[ch], sr, 6000) for ch in range(drums.shape[0])])
    toms = np.stack([bandpass(drums[ch], sr, 100, 800) for ch in range(drums.shape[0])])

    final = {
        "01_vocals": vocals,
        "02_drums": drums,
        "03_bass": bass,
        "04_piano": piano,
        "05_guitar": guitar,
        "06_other": other,
        "07_kick": kick,
        "08_snare": snare,
        "09_hihat": hihat,
        "10_toms": toms,
    }
    zip_path = os.path.join(OUT, f"{base}_10_stems.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for name, y in final.items():
            out_wav = os.path.join(OUT, f"{uid}_{name}.wav")
            # soundfile espera (samples, channels)
            if y.ndim == 2:
                y_t = y.T
            else:
                y_t = y
            sf.write(out_wav, y_t, sr)
            z.write(out_wav, arcname=f"{name}.wav")
            os.remove(out_wav)
    shutil.rmtree(os.path.join(OUT, "htdemucs_6s"), ignore_errors=True)
    os.remove(in_path)
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
