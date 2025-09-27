# Tourette's Speech Assistant

A real-time desktop application that removes stuttering and tics from microphone input for Tourette's syndrome patients, using DL models deployed on Google Cloud.

## 🎯 Overview

This application woHow to Run

There are two ways to run the app:

    Native (Linux) — simplest for development

    Docker (GUI inside container, API on host) — portable and clean

The app sends each audio block to a Scoring URL as JSON:

{ "audio_base64": "<base64 of PCM block>", "top_k": 3 }

and mutes the block if score < threshold.
0) Create a virtual “mic” sink (host)

Other apps will use the monitor of this sink as their microphone.

pactl load-module module-null-sink sink_name=VirtualMic sink_properties=device.description=VirtualMic
# later: pactl unload-module module-null-sink

    On Windows/macOS, use VB-CABLE / BlackHole instead.

1) Native (Linux)
Prereqs

    Python 3.10+

    PulseAudio or PipeWire (for audio routing)

Install & run

python3 -m venv .venv
source .venv/bin/activate
pip install PySide6 sounddevice requests numpy fastapi uvicorn

# (Optional) start a local test API that always returns {"score": 0.4}
uvicorn server:app --host 127.0.0.1 --port 8000
# leave it running in a separate terminal

# Run the GUI
python gui_mic_filter.py

In the GUI

   Input device: your real mic

   Output device: VirtualMic

   Scoring URL: http://127.0.0.1:8000/score (or your deployed endpoint)

   Threshold: 0.5 (default)

   Click Start

With the test API (score = 0.4), the Output meter ~0 (muted).

2) Docker (GUI in container, API on host)
Build the image
docker build -f Dockerfile.app -t mic-app .

Run (X11)

xhost +local:docker
docker run --rm -it \
  --network=host \
  --device /dev/snd \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $XDG_RUNTIME_DIR/pulse:/run/user/1000/pulse \
  mic-app

    Using --network=host lets the GUI call http://127.0.0.1:8000/score on your host directly.
    Wayland users: switch to the Wayland socket mount or run an Xwayland session and keep the X11 command above.

In the GUI

    Scoring URL: http://127.0.0.1:8000/score (that is for testing but to try the model use https://yamnet-service-terzmh6obq-uc.a.run.app/)

    Select Input and VirtualMic output → Start

3) Pointing to a hosted API

You don’t need to rebuild. In the GUI, change Scoring URL to your endpoint:

https://api.your-domain.com/score (https://yamnet-service-terzmh6obq-uc.a.run.app/ in this case )

(Optional) prefill via env var when running Docker:

docker run ... -e DEFAULT_SCORE_URL="https://api.your-domain.com/score" mic-app ( in this case try https://yamnet-service-terzmh6obq-uc.a.run.app/)

4) Useful knobs

    Sample rate: default 48,000 Hz (change in GUI)

    Block size: default 1024 frames (~21 ms @ 48 kHz)

    Threshold: blocks if score < threshold

    Env:

        USE_LOCAL_SCORER=1 → bypass HTTP and always use score=0.4

        USE_LOCAL_SCORER=0 (default in Dockerfile.app) → call the API

5) Verify it’s working

    Meters: Input should move when you speak; with a low score, Output ~0.

    Server logs: If using the test API, print body size to confirm requests:

    @app.post("/score")
    async def score(request: Request):
        data = await request.json()
        print("len(audio_base64) =", len(data.get("audio_base64","")))
        return {"score": 0.4}

6) Troubleshooting

    ALSA/PortAudio errors → Set 48 kHz, Block 1024–2048 in GUI.

    No devices in Docker → ensure --device /dev/snd and Pulse mount -v $XDG_RUNTIME_DIR/pulse:/run/user/1000/pulse.

    GUI doesn’t show (Docker/X11) → run xhost +local:docker and mount /tmp/.X11-unix.

    Apps can’t see the mic → choose “Monitor of VirtualMic” as microphone in the target app.
    Apps can’t see the mic → choose “Monitor of VirtualMic” as microphone in the target app.
rks as a system-wide extension to filter audio input for users with Tourette's syndrome. When the AI detects stuttering or vocal tics, it automatically removes or suppresses them from the microphone stream, allowing for clearer communication across all applications and websites.

### Key Features

- **Real-time Audio Processing**: Processes microphone input in real-time with minimal latency
- **AI-Powered Detection**: Uses YAMNet with custom classifier head for accurate tic/stutter detection
- **System-wide Integration**: Works as an extension to all applications and websites that use microphone
- **Cloud-based Inference**: Leverages Google Cloud for scalable AI processing
- **Cross-platform Desktop App**: Built with PySide6 for native desktop experience
- **Privacy-focused**: Audio processing with secure cloud integration

## 🧠 AI Architecture

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

- Custom dataset with labeled audio segments
- Binary classification: `0` (normal speech) vs `1` (tic/stutter)
- Person-based train/test split to prevent data leakage
- Data augmentation through temporal segmentation

## 🏗️ Architecture

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

## 🌐 Deployment

The application is currently deployed and accessible at:

- **Application**: Hosted at [X URL](X) 
- **Model API**: Custom classifier head available at [here](https://drive.google.com/drive/folders/1StCA0SwPgjcCzgtG_Jk8d0i4f2an49Dc?usp=sharing)

The model inference service runs on Google Cloud Platform, providing scalable and reliable AI processing for real-time audio classification.

## 🚀 Installation

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

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API access**:
   - The model is already hosted and accessible via the provided API endpoint
   - No additional cloud setup required for basic usage

4. **Run the application**:
   ```bash
   python main.py
   ```

## 🛠️ Development

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



## 📊 Performance Metrics

The model is evaluated using standard binary classification metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## 🖥️ Desktop Application

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

## 🔒 Privacy & Security

- **Minimal Data Retention**: Audio processed in real-time, not stored
- **Encrypted Communication**: All cloud API calls use HTTPS
- **Local Processing Option**: Fallback to local inference when available
- **User Control**: Full control over when processing is active

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

### Core Dependencies
```
torch>=1.9.0
tensorflow>=2.8.0
tensorflow-hub>=0.12.0
PySide6>=6.0.0
librosa>=0.9.0
soundfile>=0.10.0
google-cloud-aiplatform>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

### Development Dependencies
```
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pytest>=6.0.0
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YAMNet**: Google's YAMNet model for audio classification
- **AudioSet**: Dataset used for YAMNet pre-training
- **PySide6**: Qt for Python framework
- **Google Cloud**: AI Platform for model deployment
- **Tourette Association**: For guidance on condition understanding

## 📞 Support

For support, questions, or feature requests:
- Open an issue on GitHub
- Contact the development team
- Check the documentation wiki

---


**Note**: This application is designed to assist individuals with Tourette's syndrome but should not replace professional medical advice or treatment.



