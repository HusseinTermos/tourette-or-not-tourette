# gui_mic_filter.py
import os, sys, queue, threading, base64, io, wave
import numpy as np
import requests
import sounddevice as sd
from PySide6 import QtCore, QtWidgets

# ---------- Helpers ----------
def resample_to_16k(block: np.ndarray, sr: int) -> np.ndarray:
    """
    Linear resample to 16 kHz.
    Input: float32 PCM, shape (frames,) or (frames,channels), at sample rate sr.
    Output: same shape rank (1D stays 1D; 2D stays 2D) at 16 kHz.
    """
    TARGET = 16000
    if sr == TARGET or block.size == 0:
        return block.astype(np.float32, copy=False)

    if block.ndim == 1:
        n_in = block.shape[0]
        n_out = int(round(n_in * TARGET / sr))
        if n_out <= 0:
            return np.zeros((0,), dtype=np.float32)
        t_in  = np.linspace(0.0, 1.0, n_in,  endpoint=False, dtype=np.float32)
        t_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32)
        return np.interp(t_out, t_in, block.astype(np.float32)).astype(np.float32)

    # 2D: (frames, channels)
    n_in = block.shape[0]
    n_out = int(round(n_in * TARGET / sr))
    if n_out <= 0:
        return np.zeros((0, block.shape[1]), dtype=np.float32)
    t_in  = np.linspace(0.0, 1.0, n_in,  endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32)
    chans = [np.interp(t_out, t_in, block[:, c].astype(np.float32)).astype(np.float32)
             for c in range(block.shape[1])]
    return np.stack(chans, axis=1)

def ensure_mono(x: np.ndarray) -> np.ndarray:
    """Return mono float32 (frames,)."""
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 1:
        return x[:, 0].astype(np.float32, copy=False)
    return np.mean(x.astype(np.float32), axis=1, dtype=np.float32)

def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def wav_bytes_from_mono_float16k(sig_f32_mono: np.ndarray, sr: int = 16000) -> bytes:
    """Encode mono float32 @16k as 16-bit PCM WAV bytes."""
    pcm16 = float_to_int16(sig_f32_mono)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

# ---------- Engine ----------
class AudioEngine(QtCore.QObject):
    meters = QtCore.Signal(float, float)  # in_rms, out_rms
    xruns  = QtCore.Signal(int)
    error  = QtCore.Signal(str)

    def __init__(self, url: str, threshold: float, sr: int, block: int,
                 in_dev_index: int | None, out_dev_index: int | None, parent=None):
        super().__init__(parent)
        self.url = url
        self.threshold = threshold
        self.sr = sr
        self.block = block
        self.in_dev_index = in_dev_index
        self.out_dev_index = out_dev_index

        self._in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stop_flag = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._stream: sd.Stream | None = None
        self._xruns = 0
        self._sess = requests.Session()
        self._use_local = os.getenv("USE_LOCAL_SCORER", "0") == "1"  # default: call API

    @staticmethod
    def _caps(idx, want_input):
        d = sd.query_devices(idx) if idx is not None else sd.query_devices(sd.default.device[0 if want_input else 1])
        return d["max_input_channels"] if want_input else d["max_output_channels"]

    def _choose_channels(self):
        in_ch = self._caps(self.in_dev_index, True)
        out_ch = self._caps(self.out_dev_index, False)
        if in_ch >= 1 and out_ch >= 1:
            return 1
        if in_ch >= 2 and out_ch >= 2:
            return 2
        raise RuntimeError("Devices have no compatible channel count (need ≥1 or both ≥2).")

    def start(self):
        if self._stream is not None:
            return
        self.channels = self._choose_channels()
        dtype = "float32"

        self._stop_flag.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        def audio_cb(indata, outdata, frames, time_info, status):
            if status:
                self._xruns += 1
                self.xruns.emit(self._xruns)

            # feed worker
            chunk = indata.copy()
            if chunk.ndim == 1:
                chunk = chunk[:, None]
            try:
                self._in_q.put_nowait(chunk)
            except queue.Full:
                pass  # drop to keep RT

            # try get processed; if none, output silence
            try:
                processed = self._out_q.get_nowait()
            except queue.Empty:
                processed = np.zeros_like(indata)
            # meters
            in_rms  = float(np.sqrt(np.maximum(1e-12, np.mean(indata**2))))
            out_rms = float(np.sqrt(np.maximum(1e-12, np.mean(processed**2))))
            self.meters.emit(in_rms, out_rms)

            outdata[:] = processed

        try:
            self._stream = sd.Stream(
                samplerate=self.sr,
                blocksize=self.block,
                dtype=dtype,
                channels=self.channels,
                callback=audio_cb,
                device=(self.in_dev_index, self.out_dev_index),
            )
            self._stream.start()
        except Exception as e:
            self.error.emit(f"Audio start failed: {e}")
            self.stop()

    def stop(self):
        self._stop_flag.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=0.5)
            self._worker_thread = None
        with self._in_q.mutex:
            self._in_q.queue.clear()
        with self._out_q.mutex:
            self._out_q.queue.clear()

    def _worker_loop(self):
        url = self.url
        thr = float(self.threshold)
        timeout_s = 5.0

        # clip sizing to satisfy API: 1–3 seconds (target ~1.6 s)
        TARGET_SEC = 1.6
        MIN_SEC    = 1.1
        MAX_SEC    = 2.8

        in_sr = float(self.sr)
        min_frames    = int(MIN_SEC    * in_sr)
        target_frames = int(TARGET_SEC * in_sr)
        max_frames    = int(MAX_SEC    * in_sr)

        buf = None  # will become (0, channels) after first chunk

        while not self._stop_flag.is_set():
            # get next input chunk (frames, channels)
            try:
                chunk = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break

            if buf is None:
                buf = chunk[0:0, :]  # empty with same channel count
            buf = np.concatenate([buf, chunk], axis=0)

            # wait until at least MIN_SEC audio is accumulated
            if buf.shape[0] < min_frames:
                continue

            # aim for target; cap at max
            take = min(buf.shape[0], max_frames)
            if take < target_frames and buf.shape[0] < target_frames:
                # accumulate a bit more
                continue

            # cut a clip out of the buffer
            block = buf[:take, :]      # (frames, ch) at input SR
            buf   = buf[take:, :]      # remainder kept for next clip

            # --- Build payload: 16 kHz mono, 16-bit PCM WAV in audio_b64 ---
            mono_f32 = ensure_mono(block)               # (frames,)
            mono_16k = resample_to_16k(mono_f32, self.sr)  # 16 kHz, (frames,)
            wav_bytes = wav_bytes_from_mono_float16k(mono_16k, 16000)
            payload = {"audio_b64": base64.b64encode(wav_bytes).decode("ascii")}

            prob = 0.0
            if self._use_local:
                prob = 0.4
            else:
                try:
                    r = self._sess.post(url, json=payload, timeout=timeout_s)
                    prob = float(r.json().get("prob", 0.0))
                except Exception:
                    prob = 0.0  # fail-safe: mute

            # gate the corresponding clip to output queue (delayed by clip length)
            out_clip = block if prob >= thr else np.zeros_like(block)

            # split into stream-sized blocks for callback to drain smoothly
            bs = int(self.block)
            for i in range(0, out_clip.shape[0], bs):
                self._out_q.put(out_clip[i:i+bs, :])

# ---------- GUI ----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Mic Filter (clips 1–3s → HTTP prob → gate)")
        self.setMinimumWidth(580)

        self.in_combo  = QtWidgets.QComboBox()
        self.out_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh devices")

        self.url_edit = QtWidgets.QLineEdit(os.getenv("DEFAULT_SCORE_URL",
                                                      "https://yamnet-tic-u3i46rlgra-ww.a.run.app/predict_b64"))
        self.thr_spin = QtWidgets.QDoubleSpinBox(); self.thr_spin.setRange(0.0, 1.0); self.thr_spin.setSingleStep(0.05); self.thr_spin.setValue(0.5)
        self.sr_spin  = QtWidgets.QSpinBox(); self.sr_spin.setRange(8000, 192000); self.sr_spin.setSingleStep(1000); self.sr_spin.setValue(48000)
        self.block_spin = QtWidgets.QSpinBox(); self.block_spin.setRange(128, 8192); self.block_spin.setSingleStep(128); self.block_spin.setValue(1024)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)

        self.in_meter  = QtWidgets.QProgressBar(); self.out_meter = QtWidgets.QProgressBar()
        for m in (self.in_meter, self.out_meter): m.setRange(0, 1000)
        self.xruns_label = QtWidgets.QLabel("XRUNs: 0"); self.status_label = QtWidgets.QLabel("Idle")

        grid = QtWidgets.QGridLayout(self); r = 0
        grid.addWidget(QtWidgets.QLabel("Input device"), r,0); grid.addWidget(self.in_combo, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Output device"), r,0); grid.addWidget(self.out_combo, r,1); r+=1
        grid.addWidget(self.refresh_btn, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Scoring URL"), r,0); grid.addWidget(self.url_edit, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Threshold"), r,0); grid.addWidget(self.thr_spin, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Sample rate (I/O)"), r,0); grid.addWidget(self.sr_spin, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Block size (frames)"), r,0); grid.addWidget(self.block_spin, r,1); r+=1
        grid.addWidget(self.start_btn, r,0); grid.addWidget(self.stop_btn, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Input level"), r,0); grid.addWidget(self.in_meter, r,1); r+=1
        grid.addWidget(QtWidgets.QLabel("Output level"), r,0); grid.addWidget(self.out_meter, r,1); r+=1
        grid.addWidget(self.xruns_label, r,0); grid.addWidget(self.status_label, r,1); r+=1

        self.engine: AudioEngine | None = None
        self.refresh_btn.clicked.connect(self.populate_devices)
        self.start_btn.clicked.connect(self.start_engine)
        self.stop_btn.clicked.connect(self.stop_engine)
        self.populate_devices()

    def populate_devices(self):
        self.in_combo.clear(); self.out_combo.clear()
        devices = sd.query_devices()
        self._in_map, self._out_map = [], []
        for i, d in enumerate(devices):
            name = d["name"]
            if d["max_input_channels"] > 0:
                self._in_map.append((i, f"[{i}] {name} (in:{d['max_input_channels']})"))
            if d["max_output_channels"] > 0:
                self._out_map.append((i, f"[{i}] {name} (out:{d['max_output_channels']})"))
        for _, label in self._in_map: self.in_combo.addItem(label)
        for _, label in self._out_map: self.out_combo.addItem(label)
        for idx in range(self.out_combo.count()):
            if "virtualmic" in self.out_combo.itemText(idx).lower():
                self.out_combo.setCurrentIndex(idx); break
        self.status_label.setText("Devices loaded")

    def _current_in_index(self):
        return None if self.in_combo.currentIndex() < 0 else self._in_map[self.in_combo.currentIndex()][0]
    def _current_out_index(self):
        return None if self.out_combo.currentIndex() < 0 else self._out_map[self.out_combo.currentIndex()][0]

    def start_engine(self):
        url = self.url_edit.text().strip()
        self.engine = AudioEngine(url, float(self.thr_spin.value()), int(self.sr_spin.value()),
                                  int(self.block_spin.value()), self._current_in_index(), self._current_out_index())
        self.engine.meters.connect(self.on_meters); self.engine.xruns.connect(self.on_xruns); self.engine.error.connect(self.on_error)
        self.engine.start()
        for w in [self.in_combo, self.out_combo, self.refresh_btn, self.url_edit, self.thr_spin, self.sr_spin, self.block_spin]:
            w.setEnabled(False)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.status_label.setText("Running")

    def stop_engine(self):
        if self.engine: self.engine.stop(); self.engine = None
        self.in_meter.setValue(0); self.out_meter.setValue(0); self.xruns_label.setText("XRUNs: 0")
        for w in [self.in_combo, self.out_combo, self.refresh_btn, self.url_edit, self.thr_spin, self.sr_spin, self.block_spin]:
            w.setEnabled(True)
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.status_label.setText("Stopped")

    @QtCore.Slot(float, float)
    def on_meters(self, in_rms, out_rms):
        def scale(x): x = max(1e-6, min(1.0, x * 10.0)); return int(1000 * np.sqrt(x))
        self.in_meter.setValue(scale(in_rms)); self.out_meter.setValue(scale(out_rms))

    @QtCore.Slot(int)
    def on_xruns(self, n): self.xruns_label.setText(f"XRUNs: {n}")
    @QtCore.Slot(str)
    def on_error(self, msg): QtWidgets.QMessageBox.critical(self, "Audio Error", msg); self.stop_engine()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
