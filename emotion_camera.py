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
    from nucleo_controller import NucleoController
except ImportError:
    NucleoController = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion-mobilefacenet.onnx"
LOGO_PATH = BASE_DIR / "mfi_logo.png"
LOG_PATH = BASE_DIR / "rpi_mass.log"

FACE_CASCADE_PATH = Path(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)
EYE_CASCADE_PATH = Path(
    "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
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

# State separation tuning
TENSE_STABLE_COUNT = 2
STRESSED_STABLE_COUNT = 3
NORMAL_STABLE_COUNT = 2

# Fatigue detection
EYE_CHECK_EVERY = 3
EYE_CLOSED_TIME = 1.6
EYE_REOPEN_RESET_TIME = 0.20

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
    cv2.waitKey(100)


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
    """Show only MFI_LOGO.PNG, cropped to fill the entire screen."""
    open_fullscreen_window()

    logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_COLOR)
    if logo is None:
        log(f"Logo could not be loaded: {LOGO_PATH}")
        splash = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    else:
        splash = resize_to_fill(logo, SCREEN_WIDTH, SCREEN_HEIGHT)

    cv2.imshow(WINDOW_NAME, splash)
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )
    cv2.waitKey(2200)


def draw_panel(frame, arduino_status, emotion, state, command, confidence, fps):
    overlay = frame.copy()
    panel_width = min(370, frame.shape[1] - 20)
    panel_height = min(215, frame.shape[0] - 20)

    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    confidence_text = "---" if confidence is None else f"%{confidence * 100:.0f}"

    rows = [
        f"Nucleo  : {arduino_status}",
        f"Emotion : {emotion}",
        f"State   : {state}",
        f"Command : {command}",
        f"Confidence: {confidence_text}",
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
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not FACE_CASCADE_PATH.exists():
        raise FileNotFoundError(f"Face detector not found: {FACE_CASCADE_PATH}")

    if not EYE_CASCADE_PATH.exists():
        raise FileNotFoundError(f"Eye detector not found: {EYE_CASCADE_PATH}")

    emotion_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    emotion_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    emotion_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    face_detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    eye_detector = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))

    if face_detector.empty():
        raise RuntimeError("Haar face detector could not be opened.")

    if eye_detector.empty():
        raise RuntimeError("Haar eye detector could not be opened.")

    log("Emotion, face and eye models loaded.")
    return emotion_net, face_detector, eye_detector


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
    """
    Sharp state mapping.
    FATIGUED is NOT produced by the emotion model; eye-closure logic overrides it.
    """
    if emotion in ("ANGRY", "DISGUST"):
        return "TENSE"

    if emotion == "SAD":
        return "STRESSED"

    return "NORMAL"


def required_stable_count(state):
    if state == "TENSE":
        return TENSE_STABLE_COUNT
    if state == "STRESSED":
        return STRESSED_STABLE_COUNT
    return NORMAL_STABLE_COUNT


def connect_arduino():
    if NucleoController is None:
        log("nucleo_controller.py not found.")
        return None

    try:
        arduino = NucleoController()
        result = arduino.connect()
        if result is False:
            log("Nucleo connection failed.")
            return None
        log("Nucleo connected.")
        return arduino
    except Exception as error:
        log(f"Nucleo connection error: {error}")
        return None


def send_arduino_state(arduino, state):
    """Send exactly the same English state string that the Nucleo expects."""
    if arduino is None:
        return False

    try:
        arduino.send_command(state)
        log(f"Nucleo command sent: {state}")
        return True
    except Exception as error:
        log(f"Nucleo command error: {error}")
        return False

def close_arduino(arduino):
    if arduino is None:
        return
    try:
        close_method = getattr(arduino, "close", None)
        if callable(close_method):
            close_method()
    except Exception as error:
        log(f"Nucleo close error: {error}")


def main():
    log("RPi-MASS baslatiliyor.")
    show_splash()

    emotion_net, face_detector, eye_detector = load_models()
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

    eyes_closed_since = None
    eyes_open_since = time.time()
    eyes_currently_open = True

    shown_emotion = "WAITING"
    shown_state = "WAITING"
    shown_command = "NONE"
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

                    fatigue_active_now = (
                        eyes_closed_since is not None
                        and (time.time() - eyes_closed_since) >= EYE_CLOSED_TIME
                    )

                    if fatigue_active_now:
                        state = "FATIGUED"

                    shown_state = state

                    if state == candidate_state:
                        candidate_count += 1
                    else:
                        candidate_state = state
                        candidate_count = 1

                    now = time.time()
                    if (
                        candidate_count >= required_stable_count(state)
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

                # -------------------------------------------------
                # FATIGUE OVERRIDE - prolonged eye closure
                # -------------------------------------------------
                if frame_number % EYE_CHECK_EVERY == 0 and face_crop.size > 0:
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                    # Only search the upper ~60% of the face for eyes.
                    upper_h = max(1, int(face_gray.shape[0] * 0.62))
                    eye_roi = face_gray[:upper_h, :]

                    eyes = eye_detector.detectMultiScale(
                        eye_roi,
                        scaleFactor=1.08,
                        minNeighbors=5,
                        minSize=(18, 18)
                    )

                    now_eye = time.time()

                    if len(eyes) > 0:
                        eyes_currently_open = True
                        eyes_open_since = now_eye
                        eyes_closed_since = None
                    else:
                        eyes_currently_open = False

                        if eyes_closed_since is None:
                            eyes_closed_since = now_eye

                        closed_duration = now_eye - eyes_closed_since

                        if closed_duration >= EYE_CLOSED_TIME:
                            shown_state = "FATIGUED"
                            shown_command = "FATIGUED"

                            if last_sent_state != "FATIGUED":
                                if send_arduino_state(arduino, "FATIGUED"):
                                    last_sent_state = "FATIGUED"
                                    last_command_time = now_eye

                            candidate_state = ""
                            candidate_count = 0

                fatigue_active = (
                    eyes_closed_since is not None
                    and (time.time() - eyes_closed_since) >= EYE_CLOSED_TIME
                )

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
                shown_emotion = "FACE NOT DETECTED"
                shown_state = "WAITING"
                shown_confidence = None
                candidate_state = ""
                candidate_count = 0
                eyes_closed_since = None
                eyes_currently_open = True

            display = resize_to_fill(frame, SCREEN_WIDTH, SCREEN_HEIGHT)

            nucleo_status = "CONNECTED" if arduino is not None else "DISCONNECTED"
            draw_panel(
                display,
                nucleo_status,
                shown_emotion,
                shown_state,
                shown_command,
                shown_confidence,
                fps
            )

            cv2.imshow(WINDOW_NAME, display)
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
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
