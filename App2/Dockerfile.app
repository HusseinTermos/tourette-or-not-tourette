# Dockerfile.app — GUI-only (no API inside)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb \
    USE_LOCAL_SCORER=0

# System libs for audio + Qt GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libportaudio2 libsndfile1 alsa-utils pulseaudio-utils \
    libasound2 libasound2-plugins \
    libx11-6 libxext6 libxrender1 libxkbcommon0 libxkbcommon-x11-0 \
    libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-util1 libxcb-cursor0 libxcb-xinerama0 \
    libglib2.0-0 libgl1 libegl1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY gui_mic_filter.py /app/gui_mic_filter.py

RUN pip install --no-cache-dir PySide6 sounddevice requests numpy

CMD ["python", "gui_mic_filter.py"]
