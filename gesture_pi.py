"""
Edge AI Hand Gesture Recognition - Raspberry Pi
------------------------------------------------
Phone (IP Webcam app) -> Raspberry Pi (Teachable Machine TFLite model) -> HDMI screen

Keys (click on the video window first):
  SPACE  capture a photo, classify it, freeze it on screen (press SPACE again to go back to live)
  S      save the current frame + result into the 'captures' folder
  Q/ESC  quit
"""

import os
import time
import collections
import urllib.request

import cv2
import numpy as np

# ============ SETTINGS (edit these) ============
PHONE_IP    = "192.168.1.5"          # IP shown in the IP Webcam app
CAMERA_URL  = f"http://{PHONE_IP}:8080/shot.jpg"
MODEL_PATH  = "model.tflite"         # from Teachable Machine export
LABELS_PATH = "labels.txt"
THRESHOLD   = 0.70                   # below this -> "No gesture detected"
SMOOTHING   = 5                      # average last N live predictions (less flicker)
FULLSCREEN  = True
# ==============================================

GESTURE_TEXT = {                     # what to show on screen for each class
    "like":    "THUMBS UP  - Yes / OK",
    "dislike": "THUMBS DOWN - No",
    "fist":    "FIST",
    "palm":    "PALM - Stop",
    "peace":   "PEACE - Victory",
    "other":   "Unknown gesture",
}

# ---- load the TFLite interpreter (works with whichever package is installed) ----
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


def load_labels(path):
    # Teachable Machine format: "0 like"
    with open(path) as f:
        return [line.strip().split(" ", 1)[-1] for line in f if line.strip()]


def get_frame(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess(frame, size, dtype):
    # Same as Teachable Machine: center crop to square, resize, RGB
    h, w = frame.shape[:2]
    s = min(h, w)
    crop = frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = cv2.cvtColor(cv2.resize(crop, size), cv2.COLOR_BGR2RGB)
    if dtype == np.float32:                       # unquantized model
        img = img.astype(np.float32) / 127.5 - 1.0
    else:                                         # quantized model (uint8)
        img = img.astype(dtype)
    return np.expand_dims(img, 0)


def classify(interp, inp, out, frame):
    h, w = inp["shape"][1], inp["shape"][2]
    interp.set_tensor(inp["index"], preprocess(frame, (w, h), inp["dtype"]))
    t0 = time.perf_counter()
    interp.invoke()
    ms = (time.perf_counter() - t0) * 1000
    raw = interp.get_tensor(out["index"])[0]
    if out["dtype"] != np.float32:                # de-quantize output
        scale, zero = out["quantization"]
        raw = (raw.astype(np.float32) - zero) * (scale if scale else 1 / 255)
    return raw, ms


def draw(frame, label, conf, ms, mode):
    disp = cv2.resize(frame, (960, 720))
    ok = label is not None
    text = GESTURE_TEXT.get(label, label) if ok else "No gesture detected"
    color = (0, 200, 0) if ok and label != "other" else (0, 165, 255)

    cv2.rectangle(disp, (0, 0), (960, 110), (0, 0, 0), -1)
    cv2.putText(disp, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
    cv2.putText(disp, f"Confidence: {conf*100:.1f}%   Inference: {ms:.0f} ms   [{mode}]",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(disp, "SPACE: capture   S: save   Q: quit", (20, 705),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return disp


def main():
    labels = load_labels(LABELS_PATH)
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("Classes:", labels)
    print("Model input:", inp["shape"], inp["dtype"].__name__)

    win = "Gesture Recognition"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    history = collections.deque(maxlen=SMOOTHING)
    frozen = None
    os.makedirs("captures", exist_ok=True)

    while True:
        if frozen is None:
            try:
                frame = get_frame(CAMERA_URL)
            except Exception as e:
                print("Camera error:", e, "- check PHONE_IP and that IP Webcam is running")
                time.sleep(1)
                continue
            probs, ms = classify(interp, inp, out, frame)
            history.append(probs)
            avg = np.mean(history, axis=0)
            mode = "LIVE"
        else:
            frame, avg, ms = frozen
            mode = "CAPTURED"

        idx = int(np.argmax(avg))
        conf = float(avg[idx])
        label = labels[idx] if conf >= THRESHOLD else None
        disp = draw(frame, label, conf, ms, mode)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if frozen is None:                     # take a fresh photo and classify it alone
                try:
                    frame = get_frame(CAMERA_URL)
                    probs, ms = classify(interp, inp, out, frame)
                    frozen = (frame, probs, ms)
                    history.clear()
                except Exception as e:
                    print("Capture failed:", e)
            else:
                frozen = None
        elif key == ord("s"):
            name = f"captures/{time.strftime('%H%M%S')}_{label or 'none'}.jpg"
            cv2.imwrite(name, disp)
            print("Saved", name)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
