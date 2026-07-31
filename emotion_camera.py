#!/usr/bin/env python3
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    from arduino_controller import ArduinoController
except ImportError:
    ArduinoController = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion-mobilefacenet.onnx"
LOGO_PATH = BASE_DIR / "mfi_logo.png"
LOG_PATH = BASE_DIR / "rpi_mass.log"

FACE_CASCADE_PATH = Path(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

WINDOW_NAME = "RPi-MASS"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FACE_DETECT_EVERY = 4
EMOTION_EVERY = 8
FACE_HOLD_FRAMES = 32
STATE_STABLE_COUNT = 2
COMMAND_INTERVAL = 1.0

EMOTIONS = [
    "ANGRY",
    "DISGUST",
    "FEARFUL",
    "HAPPY",
    "NEUTRAL",
    "SAD",
    "SURPRISED",
]


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except Exception:
        pass


def get_screen_size():
    try:
        output = os.popen("xrandr --current 2>/dev/null | grep '\\*' | head -1").read()
        if output.strip():
            value = output.split()[0]
            width, height = value.split("x")
            return int(width), int(height)
    except Exception:
        pass
    return 800, 480


SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()


def resize_to_fill(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = max(target_width / w, target_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x = max(0, (new_w - target_width) // 2)
    y = max(0, (new_h - target_height) // 2)

    return resized[y:y + target_height, x:x + target_width]


def open_fullscreen_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


def centered_text(image, text, y, scale, thickness=2):
    size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(10, (image.shape[1] - size[0]) // 2)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def show_splash():
    open_fullscreen_window()

    splash = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_COLOR)

    if logo is not None:
        logo_h, logo_w = logo.shape[:2]
        max_w = int(SCREEN_WIDTH * 0.72)
        max_h = int(SCREEN_HEIGHT * 0.52)
        scale = min(max_w / logo_w, max_h / logo_h)

        logo = cv2.resize(
            logo,
            (max(1, int(logo_w * scale)), max(1, int(logo_h * scale))),
            interpolation=cv2.INTER_AREA
        )

        h, w = logo.shape[:2]
        x = (SCREEN_WIDTH - w) // 2
        y = max(15, int(SCREEN_HEIGHT * 0.05))
        splash[y:y + h, x:x + w] = logo

    centered_text(
        splash,
        "Connectivity & Smart Devices",
        int(SCREEN_HEIGHT * 0.78),
        max(0.55, SCREEN_WIDTH / 1200),
        2
    )
    centered_text(
        splash,
        "RPi-MASS",
        int(SCREEN_HEIGHT * 0.91),
        max(0.85, SCREEN_WIDTH / 750),
        3
    )

    cv2.imshow(WINDOW_NAME, splash)
    cv2.waitKey(2200)


def draw_panel(frame, arduino_status, emotion, state, command, confidence, fps):
    overlay = frame.copy()
    panel_width = min(370, frame.shape[1] - 20)
    panel_height = min(215, frame.shape[0] - 20)

    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    confidence_text = "---" if confidence is None else f"%{confidence * 100:.0f}"

    rows = [
        f"Arduino : {arduino_status}",
        f"Duygu   : {emotion}",
        f"Durum   : {state}",
        f"Komut   : {command}",
        f"Guven   : {confidence_text}",
        f"FPS     : {fps:.1f}",
    ]

    y = 42
    for row in rows:
        cv2.putText(
            frame,
            row,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += 32


def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadi: {MODEL_PATH}")

    if not FACE_CASCADE_PATH.exists():
        raise FileNotFoundError(f"Yuz modeli bulunamadi: {FACE_CASCADE_PATH}")

    emotion_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    emotion_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    emotion_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    face_detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if face_detector.empty():
        raise RuntimeError("Haar yuz modeli acilamadi.")

    log("Emotion MobileFaceNet ve yuz modeli yuklendi.")
    return emotion_net, face_detector


def open_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Kamera acilamadi.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    log("Kamera acildi.")
    return camera


def expand_face_box(box, image_shape, margin_x=0.18, margin_y=0.22):
    x, y, w, h = [int(v) for v in box]
    image_h, image_w = image_shape[:2]

    mx = int(w * margin_x)
    my = int(h * margin_y)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(image_w, x + w + mx)
    y2 = min(image_h, y + h + my)

    return x1, y1, x2, y2


def infer_emotion(net, face_bgr):
    """
    OpenCV Zoo model preprocessing:
    BGR -> RGB, 112x112, [0,1], then (x - 0.5) / 0.5.
    The official model generally returns a class label named 'label'.
    """
    if face_bgr is None or face_bgr.size == 0:
        return "NEUTRAL", None

    face = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    blob = cv2.dnn.blobFromImage(face)

    net.setInput(blob, "data")

    try:
        output = net.forward(["label"])[0]
    except cv2.error:
        output = net.forward()

    values = np.asarray(output).reshape(-1)

    if values.size == 1:
        index = int(round(float(values[0])))
        confidence = None
    else:
        shifted = values.astype(np.float32) - np.max(values)
        probabilities = np.exp(shifted)
        probabilities /= max(float(np.sum(probabilities)), 1e-9)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

    if not 0 <= index < len(EMOTIONS):
        return "NEUTRAL", confidence

    return EMOTIONS[index], confidence


def emotion_to_state(emotion):
    if emotion in ("ANGRY", "DISGUST", "FEARFUL"):
        return "GERGIN"
    if emotion == "SAD":
        return "STRESLI"
    return "NORMAL"


def connect_arduino():
    if ArduinoController is None:
        log("arduino_controller.py bulunamadi.")
        return None

    try:
        arduino = ArduinoController()
        result = arduino.connect()
        if result is False:
            log("Arduino baglanamadi.")
            return None
        log("Arduino baglandi.")
        return arduino
    except Exception as error:
        log(f"Arduino baglanti hatasi: {error}")
        return None


def call_first_available(obj, method_names):
    for name in method_names:
        method = getattr(obj, name, None)
        if callable(method):
            method()
            return True
    return False


def send_arduino_state(arduino, state):
    if arduino is None:
        return False

    try:
        method_map = {
            "NORMAL": ("normal",),
            "GERGIN": ("tense", "gergin"),
            "STRESLI": ("stressed", "stresli"),
            "YORGUN": ("fatigue", "yorgun"),
        }

        if call_first_available(arduino, method_map.get(state, ("normal",))):
            log(f"Arduino komutu gonderildi: {state}")
            return True

        generic = getattr(arduino, "send_command", None)
        if callable(generic):
            generic(state)
            log(f"Arduino komutu gonderildi: {state}")
            return True

        serial_obj = getattr(arduino, "serial", None)
        if serial_obj is not None and hasattr(serial_obj, "write"):
            serial_obj.write((state + "\n").encode("utf-8"))
            log(f"Arduino komutu gonderildi: {state}")
            return True

        log("ArduinoController icinde uygun komut metodu bulunamadi.")
        return False

    except Exception as error:
        log(f"Arduino komut hatasi: {error}")
        return False


def close_arduino(arduino):
    if arduino is None:
        return
    try:
        close_method = getattr(arduino, "close", None)
        if callable(close_method):
            close_method()
    except Exception as error:
        log(f"Arduino kapatma hatasi: {error}")


def main():
    log("RPi-MASS baslatiliyor.")
    show_splash()

    emotion_net, face_detector = load_models()
    camera = open_camera()
    arduino = connect_arduino()

    open_fullscreen_window()

    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    frame_number = 0
    fps_count = 0
    fps = 0.0
    fps_time = time.time()

    last_face = None
    no_face_frames = FACE_HOLD_FRAMES + 1

    shown_emotion = "BEKLEME"
    shown_state = "BEKLEME"
    shown_command = "HENUZ YOK"
    shown_confidence = None

    candidate_state = ""
    candidate_count = 0
    last_sent_state = ""
    last_command_time = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Kameradan goruntu alinamadi.")

            frame_number += 1
            fps_count += 1

            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()

            frame = cv2.flip(frame, 1)

            if frame_number % FACE_DETECT_EVERY == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                small = cv2.resize(
                    gray, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA
                )

                faces = face_detector.detectMultiScale(
                    small,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(50, 50),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                if len(faces) > 0:
                    face = max(faces, key=lambda item: item[2] * item[3])
                    scale_back = 1.0 / 0.65
                    last_face = tuple(int(v * scale_back) for v in face)
                    no_face_frames = 0
                else:
                    no_face_frames += FACE_DETECT_EVERY
                    if no_face_frames > FACE_HOLD_FRAMES:
                        last_face = None

            if future is not None and future.done():
                try:
                    emotion, confidence = future.result()
                    shown_emotion = emotion
                    shown_confidence = confidence
                    state = emotion_to_state(emotion)
                    shown_state = state

                    if state == candidate_state:
                        candidate_count += 1
                    else:
                        candidate_state = state
                        candidate_count = 1

                    now = time.time()
                    if (
                        candidate_count >= STATE_STABLE_COUNT
                        and state != last_sent_state
                        and now - last_command_time >= COMMAND_INTERVAL
                    ):
                        if send_arduino_state(arduino, state):
                            last_sent_state = state
                            last_command_time = now
                        shown_command = state
                except Exception as error:
                    log(f"Duygu analiz hatasi: {error}")
                future = None

            if last_face is not None:
                x, y, w, h = last_face
                x1, y1, x2, y2 = expand_face_box(last_face, frame.shape)
                face_crop = frame[y1:y2, x1:x2]

                if (
                    frame_number % EMOTION_EVERY == 0
                    and future is None
                    and face_crop.size > 0
                ):
                    future = executor.submit(
                        infer_emotion, emotion_net, face_crop.copy()
                    )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2
                )
            else:
                shown_emotion = "YUZ YOK"
                shown_state = "BEKLEME"
                shown_confidence = None
                candidate_state = ""
                candidate_count = 0

            display = resize_to_fill(frame, SCREEN_WIDTH, SCREEN_HEIGHT)

            arduino_status = "BAGLI" if arduino is not None else "BAGLI DEGIL"
            draw_panel(
                display,
                arduino_status,
                shown_emotion,
                shown_state,
                shown_command,
                shown_confidence,
                fps
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

    finally:
        if future is not None and not future.done():
            future.cancel()

        executor.shutdown(wait=False, cancel_futures=True)
        close_arduino(arduino)
        camera.release()
        cv2.destroyAllWindows()
        log("RPi-MASS kapatildi.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        log(error)
        print(error)
        sys.exit(1)
