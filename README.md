<div align="center">

# Tourette's Speech Assistant

Empowering clearer speech: real-time filtering of vocal tics and stutter events using an AI audio classifier.

</div>

---

## TL;DR
Desktop app that captures microphone audio, sends short chunks to a hosted AI scoring endpoint (YAMNet + custom head), and suppresses blocks likely containing tic/stutter artifacts before forwarding them to a virtual microphone that other apps (Zoom, Meet, Discord, etc.) can use.

Hosted model endpoint: `https://yamnet-service-terzmh6obq-uc.a.run.app/score`

---

##  Overview
Our project reduces audible tics and disruptive stutter bursts in live voice communication. It operates as a system-level filtering layer: your real mic → classifier → cleaned stream → virtual mic consumed by other applications.

Core goals:
- Lower social friction for users with Tourette's & speech disfluencies
- Maintain very low latency
- Provide transparent controls & local fallback options

## Value Proposition
- **Speak Freely & Confidently**: Reduce the self-consciousness associated with vocal tics and stutters in online meetings, chats, and recordings.
- **Seamless Integration**: Works automatically with all your favorite apps (Zoom, Discord, Google Meet, etc.) with a one-time setup.
- **User-Controlled & Private**: You control the filtering sensitivity. Your audio is processed in real-time and never stored.

## Features
- Near real-time streaming classification
- YAMNet embeddings + lightweight PyTorch classifier head
- Adjustable suppression threshold
- Works with any app that can select an input device
- Virtual microphone routing (PulseAudio/PipeWire sink)
- Hosted API + optional local dummy scorer
- Privacy-friendly: no long-term storage of audio

## Model & Data
| Component | Details |
|-----------|---------|
| Backbone  | YAMNet (TF Hub) frozen, 1024‑D embeddings |
| Head      | Dropout → Linear(256) → ReLU → Dropout → Linear(1) (sigmoid at inference) |
| Task      | Binary classification (tic/stutter = 1, normal = 0) |
| Windowing | 2s segments, 0.5s hop (training/augmentation) |
| Inference Blocks | ~20–40 ms PCM chunks aggregated into score smoothing |

### Dataset
We manually assembled a custom dataset of 120+ labeled audio samples (positive tic/stutter vs negative normal speech) sourced from publicly available social media clips. Steps taken:
- Curated short speech segments with and without vocal tics
- Normalized to mono 16 kHz
- Segmented with sliding windows for augmentation
- Person-based split to avoid speaker leakage

Limitations:
- Small dataset size (still exploratory)
- Social media audio may include compression artifacts
- Model may misclassify atypical speech prosody

## System Architecture
```
┌──────────────┐   ┌──────────────┐   ┌────────────────────┐   ┌───────────────────┐
│  Microphone  │→→│  GUI Capture  │→→│  Scoring Endpoint   │→→│  Suppression Logic │
└──────────────┘   │ (PySide6)    │   │  (Hosted API)      │   └─────────┬─────────┘
               └──────┬──────┘   └────────────────────┘             │
                    │                                            │ filtered
                    ▼                                            ▼
               ┌──────────────┐                             ┌──────────────┐
               │ Virtual Sink │→→ Apps (Zoom/Meet/etc.)     │ User Monitor │
               └──────────────┘                             └──────────────┘
```

## Installation (Linux Dev Environment)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # (create one if not present)
```
If you don't yet have a `requirements.txt`, a minimal starter:
```txt
PySide6
sounddevice
requests
numpy
fastapi
uvicorn
torch
tensorflow
tensorflow-hub
```

### Create Virtual Microphone (PulseAudio / PipeWire)
```bash
pactl load-module module-null-sink sink_name=VirtualMic sink_properties=device.description=VirtualMic
# To remove later:

```
On Windows/macOS use VB-CABLE or BlackHole.

## Running the App
### Local test scorer just to make sure the app is running 
```bash
uvicorn server:app --host 127.0.0.1 --port 8000
python gui_mic_filter.py
```
Then set Test Scoring URL(change this after): `http://127.0.0.1:8000/score`.

### Option 2: Hosted model
Set Scoring URL: `https://yamnet-tic-u3i46rlgra-ww.a.run.app/`

### GUI Checklist
1. Input Device = your real microphone
2. Output Device = VirtualMic
3. Threshold = start around 0.5
4. Click Start → watch Input vs Output meters

## API Contract
Endpoint (POST): `/score`
```jsonc
{
  "audio_b64": "<base64 PCM int16 or float32 mono>",
  
}
```
Successful response:
```json
{ "score": 0.42 }
```
Interpretation: lower score → more suppression (depending on inversion logic you implement). Adjust threshold to taste.

### Example (Python client snippet)
```python
import base64, requests, sounddevice as sd, numpy as np

def record_block(seconds=0.3, sr=48000):
   audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype='float32')
   sd.wait()
   return audio.squeeze()

blk = record_block()
payload = {
   "audio_b64": base64.b64encode(blk.tobytes()).decode(),
   
}
r = requests.post("https://yamnet-tic-u3i46rlgra-ww.a.run.app/", json=payload, timeout=5)
print(r.json())
```

##  Development Notes
Training logic currently lives in notebooks (`model2.ipynb`, `termos_data.ipynb`). Consider exporting core model code into a module for reproducibility. Future refactor suggestion:
```
src/
  data/
  models/
  training/
  gui/
```

## Metrics & Evaluation
The model was evaluated on a held-out test set of speakers not seen during training.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.7778 |
| Precision | 0.8095 |
| Recall    | 0.8947 |
| F1-Score  | 0.8500 |


## Future Roadmap
- [ ] Add confidence smoothing (EMA over last N blocks)
- [ ] Implement per-speaker calibration allowing them to specify words that that they usually say in tics 
- [ ] Export ONNX / TorchScript for lighter inference
- [ ] Add Windows/macOS virtual mic setup docs
- [ ] Provide packaged installers (.deb / .msi)
- [ ] Add logging & opt‑in telemetry
- [ ] Collect more diverse tic types (ethical sourcing + consent)

## Docker (Optional)
```bash
docker build -f Dockerfile.app -t mic-app .
xhost +local:docker
docker run --rm -it \
  --network=host \
  --device /dev/snd \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $XDG_RUNTIME_DIR/pulse:/run/user/1000/pulse \
  mic-app
```
If using Wayland, adapt DISPLAY/socket mounts accordingly.

##  Troubleshooting
| Issue | Suggestion |
|-------|------------|
| No devices listed | Check PulseAudio / PipeWire is running, permissions, container flags |
| Output always muted | Threshold too high or scorer always returns low values |
| High latency | Reduce block size (trade-off: more HTTP calls) |
| Distortion | Ensure consistent sample rate (48k GUI vs 16k model) |
| API timeouts | Add retry/backoff, verify endpoint health |

##  Privacy & Ethics
- Streaming only; no persistent storage by default
- Social media sourced dataset: avoid redistribution without rights
- Consider adding an in-app consent & transparency panel
- Provide a local inference option for sensitive environments

##  Contributing
```bash
git checkout -b feature/your-feature
# make changes
git commit -m "feat: add your feature"
git push origin feature/your-feature
```
Open a PR describing motivation + screenshots (if UI change).

##  License
MIT (see `LICENSE`).

##  Acknowledgments
YAMNet (Google), AudioSet, PySide6, open-source community, and individuals who publicly share educational content about Tourette's (ethical use only).

##  Disclaimer
Not a medical device. Does not treat or diagnose. Always consult healthcare professionals for clinical needs.

---
Made with focus on accessibility and dignity.
##  AI Architecture

### Model Components

1. **YAMNet Backbone**: Google's YAMNet (Yet Another Mobile Network) pre-trained audio classification model
   - Provides robust 1024-dimensional audio embeddings
   - Trained on AudioSet dataset for general audio understanding
   - Handles 16kHz mono audio input

2. **Custom Classifier Head**: Binary classification layer for Tourette's detection
   ```
   YAMNet Embeddings (1024D) → Dropout → Linear(256) → ReLU → Dropout → Linear(1) → Sigmoid
   ```

3. **Audio Preprocessing Pipeline**:
   - Sliding window segmentation (2-second windows, 0.5-second hop)
   - Sample rate normalization to 16kHz

### Training Data

- **Custom Dataset**: We gathered our own dataset consisting of 120+ samples of both positive (tic/stutter) and negative (normal speech) audio samples, collected from social media platforms.
- **Balanced Classification**: Binary classification with `0` (normal speech) vs `1` (tic/stutter) labels
- **Diverse Sample Collection**: Audio samples collected from multiple speakers to ensure model generalization
- **Person-based train/test split**: Prevents data leakage by ensuring no speaker appears in both training and testing sets
- **Data augmentation**: Applied temporal segmentation through sliding windows to increase dataset size

##  Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│   PySide6 App    │───▶│  Google Cloud   │
│     Input       │    │  (Real-time      │    │   AI Service    │
└─────────────────┘    │   Processing)    │    └─────────────────┘
                       └──────────────────┘              │
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Applications  │◀───│  Filtered Audio  │◀───│  YAMNet Model   │
│   & Websites    │    │     Output       │    │  + Classifier   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

##  Deployment

The application is currently deployed and accessible at:

- **Application**: Hosted at [X URL](X) 
- **Model API**: Custom classifier head available at [here](https://drive.google.com/drive/folders/1StCA0SwPgjcCzgtG_Jk8d0i4f2an49Dc?usp=sharing)

The model inference service runs on Google Cloud Platform, providing scalable and reliable AI processing for real-time audio classification.

##  Installation

### Prerequisites

- Python 3.8+
- PySide6
- PyTorch
- TensorFlow


### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HusseinTermos/tourette-or-not-tourette.git
   cd tourette-or-not-tourette
   ```
2. (Linux) install system libs
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  libportaudio2 libsndfile1 libasound2 \
  libx11-6 libxext6 libxrender1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-util1 libxcb-cursor0 libxcb-xinerama0 \
  libgl1 libegl1
# If sounddevice build fails: sudo apt install -y portaudio19-dev

3) Create a virtual mic (host)

Other apps (e.g., Google Meet) will use the monitor of this sink as their microphone.

pactl load-module module-null-sink sink_name=VirtualMic sink_properties=device.description=VirtualMic
# to remove later: pactl unload-module module-null-sink

4) Create a venv & install Python deps

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

5) Run the GUI app

python gui_mic_filter.py

In the GUI:

    Input device: your real mic

    Output device: VirtualMic

    Scoring URL: https://yamnet-tic-u3i46rlgra-ww.a.run.app/predict_b64

    Threshold: 0.5

    Click Start.

What the app sends

    One request every ~1.6 s

    JSON body: {"audio_b64": "<base64 of mono 16 kHz 16-bit PCM WAV>"}

    Audio shorter than 1 s or longer than 3 s is not sent (the app batches ~1.6 s by design).

6) Use in Google Meet

    Open Meet → Settings → Audio.

    Set Microphone to “Monitor of VirtualMic”.

        If not visible, create a remapped source:

pactl load-module module-remap-source master=VirtualMic.monitor source_name=VirtualMicMic source_properties=device.description=VirtualMic

Then select VirtualMic (or VirtualMicMic) in Meet.

Option 2 use Docker 
Docker (Linux)

Run the GUI in a container and route its output to a virtual mic your browser can pick (e.g., for Google Meet).

1) Build the image
# from repo root
docker build -f Dockerfile.app -t mic-app .

2) Create a virtual mic on the host

Other apps will use the monitor of this sink as their microphone.

pactl load-module module-null-sink sink_name=VirtualMic \
  sink_properties=device.description=VirtualMic
# to remove later: pactl unload-module module-null-sink

3) Run the container (X11)

If your API is already hosted (recommended), set the URL and run:

# allow X clients from docker
xhost +local:docker

docker run --rm -it \
  --device /dev/snd \
  -e DISPLAY=$DISPLAY \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -e DEFAULT_SCORE_URL="https://yamnet-tic-u3i46rlgra-ww.a.run.app/predict_b64" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $XDG_RUNTIME_DIR/pulse:/run/user/1000/pulse \
  mic-app
3. **Configure API access**:
   - The model is already hosted and accessible via the provided API endpoint
   - No additional cloud setup required for basic usage
   

##  Development

### Model Training

The model training pipeline is available in Jupyter notebooks:

- `model2.ipynb`: Main model training and evaluation
- `termos_data.ipynb`: Data preprocessing and augmentation

#### Training Process

1. **Data Preparation**:
   ```python
   # Load and segment audio files
   # Apply sliding window (2s window, 0.5s hop)
   # Person-based train/test split
   ```

2. **Model Architecture**:
   ```python
   class TicClassifier(nn.Module):
       def __init__(self):
           self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
           self.classifier = nn.Sequential(...)
   ```

3. **Training Loop**:
   - Binary cross-entropy loss
   - AdamW optimizer
   - Person-based validation to prevent overfitting



##  Performance Metrics

The model is evaluated using standard binary classification metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

##  Desktop Application

### PySide6 Interface

The desktop application provides:

- **System Tray Integration**: Runs quietly in the background
- **Real-time Status**: Visual indicators for processing status
- **Settings Panel**: Adjust sensitivity and processing parameters
- **Audio Device Selection**: Choose input/output devices
- **Privacy Controls**: Local vs cloud processing options

### Key Components

- **Audio Capture**: Real-time microphone input handling
- **Cloud Communication**: Secure API calls to Google Cloud
- **Audio Playback**: Filtered audio output to system
- **UI Management**: PySide6-based user interface



