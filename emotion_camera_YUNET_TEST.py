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

# Accept either the old short FER filename or the official OpenCV Zoo filename.
FER_CANDIDATES = [
    BASE_DIR / "emotion-mobilefacenet.onnx",
    BASE_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx",
]
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
LOGO_PATH = BASE_DIR / "mfi_logo.png"
LOG_PATH = BASE_DIR / "rpi_mass.log"

WINDOW_NAME = "RPi-MASS YUNET TEST"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Performance / stability
FACE_DETECT_EVERY = 2
EMOTION_EVERY = 6
STATE_STABLE_COUNT = 3
COMMAND_INTERVAL = 1.0

# Fatigue: eyes closed continuously for this long.
FATIGUE_SECONDS = 1.6

# YuNet
YUNET_SCORE_THRESHOLD = 0.72
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

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
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_screen_size():
    try:
        output = os.popen("xrandr --current 2>/dev/null | grep '\\*' | head -1").read()
        if output.strip():
            value = output.split()[0]
            w, h = value.split("x")
            return int(w), int(h)
    except Exception:
        pass
    return 800, 480


SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()


def resize_to_fill(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = max(target_width / w, target_height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = max(0, (nw - target_width) // 2)
    y = max(0, (nh - target_height) // 2)
    return resized[y:y + target_height, x:x + target_width]


def open_fullscreen_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)


def show_splash():
    open_fullscreen_window()
    logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_COLOR)
    if logo is None:
        splash = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        log(f"Logo could not be loaded: {LOGO_PATH}")
    else:
        splash = resize_to_fill(logo, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.imshow(WINDOW_NAME, splash)
    cv2.waitKey(1800)


def find_fer_model():
    for p in FER_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "FER model not found. Expected emotion-mobilefacenet.onnx or "
        "facial_expression_recognition_mobilefacenet_2022july.onnx"
    )


def load_models():
    fer_path = find_fer_model()
    if not YUNET_PATH.exists():
        raise FileNotFoundError(f"YuNet model not found: {YUNET_PATH}")

    fer_net = cv2.dnn.readNetFromONNX(str(fer_path))
    fer_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    fer_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    yunet = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K,
    )

    log(f"FER model: {fer_path.name}")
    log(f"YuNet model: {YUNET_PATH.name}")
    return fer_net, yunet


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# Standard 112x112 five-point face template (ArcFace-style).
DST_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def yunet_landmarks(face):
    # YuNet row: x,y,w,h, right-eye?, left-eye?, nose, mouth corners, score.
    # OpenCV Zoo ordering is five 2D landmarks after bbox.
    pts = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)

    # Ensure first two points are ordered left-to-right in image coordinates
    # for the alignment template.
    if pts[0, 0] > pts[1, 0]:
        pts[[0, 1]] = pts[[1, 0]]

    # Mouth corners also left-to-right.
    if pts[3, 0] > pts[4, 0]:
        pts[[3, 4]] = pts[[4, 3]]

    return pts


def align_face(frame, face):
    src_pts = yunet_landmarks(face)
    matrix, _ = cv2.estimateAffinePartial2D(
        src_pts,
        DST_5PTS,
        method=cv2.LMEDS,
    )
    if matrix is None:
        return None

    aligned = cv2.warpAffine(
        frame,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def infer_emotion(net, aligned_bgr):
    if aligned_bgr is None or aligned_bgr.size == 0:
        return "NEUTRAL", None

    # Keep the preprocessing used by the existing working model.
    face = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
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
        idx = int(round(float(values[0])))
        confidence = None
    else:
        shifted = values.astype(np.float32) - np.max(values)
        probs = np.exp(shifted)
        probs /= max(float(np.sum(probs)), 1e-9)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

    if not 0 <= idx < len(EMOTIONS):
        return "NEUTRAL", confidence
    return EMOTIONS[idx], confidence


def expression_to_state(emotion):
    # Keep states intentionally separated.
    if emotion in ("ANGRY", "DISGUST"):
        return "TENSE"
    if emotion == "SAD":
        return "STRESSED"
    return "NORMAL"


def eye_openness_ratio(face):
    """
    Geometric fatigue cue from YuNet landmarks.
    This does not classify emotion. It estimates how far the eye landmarks sit
    above the nose relative to face height. We calibrate an open-eye baseline
    during runtime and detect a sustained drop from that baseline.
    """
    pts = yunet_landmarks(face)
    eye_mid = (pts[0] + pts[1]) * 0.5
    nose = pts[2]
    h = max(float(face[3]), 1.0)
    return float((nose[1] - eye_mid[1]) / h)


def connect_nucleo():
    if NucleoController is None:
        log("nucleo_controller.py not found.")
        return None
    try:
        nucleo = NucleoController()
        if nucleo.connect() is False:
            return None
        log("Nucleo connected.")
        return nucleo
    except Exception as e:
        log(f"Nucleo connection error: {e}")
        return None


def send_state(nucleo, state):
    if nucleo is None:
        return False
    try:
        nucleo.send_command(state)
        log(f"Nucleo command sent: {state}")
        return True
    except Exception as e:
        log(f"Nucleo command error: {e}")
        return False


def draw_panel(frame, connected, emotion, state, command, confidence, fps, eye_text):
    overlay = frame.copy()
    panel_w = min(390, frame.shape[1] - 20)
    panel_h = min(245, frame.shape[0] - 20)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    conf = "---" if confidence is None else f"{confidence * 100:.0f}%"
    rows = [
        f"Nucleo     : {'CONNECTED' if connected else 'DISCONNECTED'}",
        f"Emotion    : {emotion}",
        f"State      : {state}",
        f"Command    : {command}",
        f"Confidence : {conf}",
        f"Eyes       : {eye_text}",
        f"FPS        : {fps:.1f}",
    ]
    y = 38
    for row in rows:
        cv2.putText(frame, row, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 29


def main():
    log("YuNet test starting.")
    show_splash()

    fer_net, yunet = load_models()
    cap = open_camera()
    nucleo = connect_nucleo()
    open_fullscreen_window()

    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    frame_no = 0
    fps = 0.0
    fps_count = 0
    fps_t = time.time()

    last_face = None
    shown_emotion = "WAITING"
    shown_state = "WAITING"
    shown_command = "NONE"
    shown_conf = None

    candidate = ""
    candidate_count = 0
    last_sent = ""
    last_send_t = 0.0

    # Runtime eye baseline.
    eye_baseline = None
    closed_since = None
    eye_text = "CALIBRATING"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame could not be read.")

            frame = cv2.flip(frame, 1)
            frame_no += 1
            fps_count += 1

            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_count / (now - fps_t)
                fps_count = 0
                fps_t = now

            if frame_no % FACE_DETECT_EVERY == 0:
                h, w = frame.shape[:2]
                yunet.setInputSize((w, h))
                _, faces = yunet.detect(frame)
                if faces is not None and len(faces):
                    last_face = max(faces, key=lambda f: float(f[2] * f[3])).copy()
                else:
                    last_face = None

            fatigue_active = False

            if last_face is not None:
                x, y, w, h = [int(v) for v in last_face[:4]]
                x = max(0, x)
                y = max(0, y)
                w = max(1, min(w, frame.shape[1] - x))
                h = max(1, min(h, frame.shape[0] - y))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Draw YuNet landmarks for this test version.
                pts = yunet_landmarks(last_face)
                for px, py in pts:
                    cv2.circle(frame, (int(px), int(py)), 3, (255, 255, 255), -1)

                ratio = eye_openness_ratio(last_face)

                # Build a slow baseline from normal/open-looking frames.
                if eye_baseline is None:
                    eye_baseline = ratio
                elif closed_since is None:
                    eye_baseline = 0.98 * eye_baseline + 0.02 * ratio

                # A substantial reduction relative to personal baseline.
                threshold = eye_baseline * 0.82
                eyes_closed = ratio < threshold

                if eyes_closed:
                    if closed_since is None:
                        closed_since = now
                    closed_for = now - closed_since
                    eye_text = f"CLOSED {closed_for:.1f}s"
                    fatigue_active = closed_for >= FATIGUE_SECONDS
                else:
                    closed_since = None
                    eye_text = "OPEN"

                if future is None and frame_no % EMOTION_EVERY == 0:
                    aligned = align_face(frame, last_face)
                    if aligned is not None:
                        future = executor.submit(infer_emotion, fer_net, aligned.copy())

            else:
                closed_since = None
                eye_text = "NO FACE"
                shown_emotion = "FACE NOT DETECTED"
                shown_state = "WAITING"
                candidate = ""
                candidate_count = 0

            if future is not None and future.done():
                try:
                    emotion, conf = future.result()
                    shown_emotion = emotion
                    shown_conf = conf

                    state = expression_to_state(emotion)

                    if fatigue_active:
                        state = "FATIGUED"

                    if state == candidate:
                        candidate_count += 1
                    else:
                        candidate = state
                        candidate_count = 1

                    shown_state = state

                    # FATIGUED can react immediately after the sustained eye timer.
                    required = 1 if state == "FATIGUED" else STATE_STABLE_COUNT

                    if (candidate_count >= required
                            and state != last_sent
                            and now - last_send_t >= COMMAND_INTERVAL):
                        if send_state(nucleo, state):
                            last_sent = state
                            last_send_t = now
                        shown_command = state

                except Exception as e:
                    log(f"Inference error: {e}")
                future = None

            # Important: fatigue override must also work between emotion inferences.
            if fatigue_active:
                shown_state = "FATIGUED"
                if last_sent != "FATIGUED" and now - last_send_t >= COMMAND_INTERVAL:
                    if send_state(nucleo, "FATIGUED"):
                        last_sent = "FATIGUED"
                        last_send_t = now
                    shown_command = "FATIGUED"

            display = resize_to_fill(frame, SCREEN_WIDTH, SCREEN_HEIGHT)
            draw_panel(
                display,
                nucleo is not None,
                shown_emotion,
                shown_state,
                shown_command,
                shown_conf,
                fps,
                eye_text,
            )

            cv2.imshow(WINDOW_NAME, display)
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        if future is not None and not future.done():
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

        try:
            if nucleo is not None:
                nucleo.close()
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        log("YuNet test stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        log(err)
        print(err)
        sys.exit(1)
