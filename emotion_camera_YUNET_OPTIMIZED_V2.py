#!/usr/bin/env python3
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from nucleo_controller import NucleoController
except ImportError:
    NucleoController = None


BASE_DIR = Path(__file__).resolve().parent

FER_CANDIDATES = [
    BASE_DIR / "emotion-mobilefacenet.onnx",
    BASE_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx",
]

YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
LOGO_PATH = BASE_DIR / "mfi_logo.png"
LOG_PATH = BASE_DIR / "rpi_mass.log"

EYE_CASCADE_PATH = Path(
    "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
)

WINDOW_NAME = "RPi-MASS"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---------------------------------------------------------
# PERFORMANCE / STABILITY
# ---------------------------------------------------------

# YuNet and FER are intentionally not executed on every frame.
FACE_DETECT_EVERY = 6
EMOTION_EVERY = 12
EYE_CHECK_EVERY = 4

COMMAND_INTERVAL = 0.8
COMMAND_REFRESH_INTERVAL = 3.0

# State must be observed this many consecutive emotion results.
STATE_CONFIRM_COUNT = 2

# Fatigue is independent from emotion classification.
# We use a rolling eye-status window + hysteresis to avoid false fatigue.
FATIGUE_SECONDS = 2.8
FATIGUE_RECOVERY_SECONDS = 1.2
EYE_HISTORY_SIZE = 8
FATIGUE_ENTER_OPEN_RATIO = 0.25
FATIGUE_EXIT_OPEN_RATIO = 0.65

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
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except Exception:
        pass


def get_screen_size():
    try:
        output = os.popen(
            "xrandr --current 2>/dev/null | grep '\\*' | head -1"
        ).read()

        if output.strip():
            value = output.split()[0]
            width, height = value.split("x")
            return int(width), int(height)

    except Exception:
        pass

    return 800, 480


SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()


def resize_to_fill(image, target_width, target_height):
    image_height, image_width = image.shape[:2]

    scale = max(
        target_width / image_width,
        target_height / image_height,
    )

    new_width = max(
        1,
        int(image_width * scale),
    )

    new_height = max(
        1,
        int(image_height * scale),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    start_x = max(
        0,
        (new_width - target_width) // 2,
    )

    start_y = max(
        0,
        (new_height - target_height) // 2,
    )

    return resized[
        start_y:start_y + target_height,
        start_x:start_x + target_width,
    ]


def open_fullscreen_window():
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.moveWindow(
        WINDOW_NAME,
        0,
        0,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    cv2.waitKey(100)


def show_splash():
    open_fullscreen_window()

    logo = cv2.imread(
        str(LOGO_PATH),
        cv2.IMREAD_COLOR,
    )

    if logo is None:
        log(
            f"Logo could not be loaded: {LOGO_PATH}"
        )

        splash = np.zeros(
            (
                SCREEN_HEIGHT,
                SCREEN_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

    else:
        splash = resize_to_fill(
            logo,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

    cv2.imshow(
        WINDOW_NAME,
        splash,
    )

    cv2.waitKey(1800)


def find_fer_model():
    for path in FER_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "FER model not found."
    )


def load_models():
    fer_path = find_fer_model()

    if not YUNET_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_PATH}"
        )

    if not EYE_CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Eye detector not found: {EYE_CASCADE_PATH}"
        )

    fer_net = cv2.dnn.readNetFromONNX(
        str(fer_path)
    )

    fer_net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    fer_net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    yunet = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        ),
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K,
    )

    eye_detector = cv2.CascadeClassifier(
        str(EYE_CASCADE_PATH)
    )

    if eye_detector.empty():
        raise RuntimeError(
            "Eye detector could not be opened."
        )

    log(
        f"FER model loaded: {fer_path.name}"
    )

    log(
        f"YuNet model loaded: {YUNET_PATH.name}"
    )

    return (
        fer_net,
        yunet,
        eye_detector,
    )


def open_camera():
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():
        raise RuntimeError(
            "Camera could not be opened."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    log("Camera opened.")
    return camera


# ---------------------------------------------------------
# YUNET FACE ALIGNMENT
# ---------------------------------------------------------

DST_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def yunet_landmarks(face):
    points = np.asarray(
        face[4:14],
        dtype=np.float32,
    ).reshape(5, 2)

    if points[0, 0] > points[1, 0]:
        points[[0, 1]] = points[[1, 0]]

    if points[3, 0] > points[4, 0]:
        points[[3, 4]] = points[[4, 3]]

    return points


def align_face(frame, face):
    source_points = yunet_landmarks(
        face
    )

    matrix, _ = cv2.estimateAffinePartial2D(
        source_points,
        DST_5PTS,
        method=cv2.LMEDS,
    )

    if matrix is None:
        return None

    return cv2.warpAffine(
        frame,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------
# BACKGROUND FACE DETECTION
# ---------------------------------------------------------

def detect_face_yunet(yunet, frame):
    height, width = frame.shape[:2]

    yunet.setInputSize(
        (width, height)
    )

    _, faces = yunet.detect(
        frame
    )

    if faces is None or len(faces) == 0:
        return None

    largest = max(
        faces,
        key=lambda face:
        float(face[2] * face[3]),
    )

    return largest.copy()


# ---------------------------------------------------------
# EMOTION INFERENCE
# ---------------------------------------------------------

def infer_emotion(net, aligned_bgr):
    if (
        aligned_bgr is None
        or aligned_bgr.size == 0
    ):
        return "NEUTRAL", None

    face = cv2.cvtColor(
        aligned_bgr,
        cv2.COLOR_BGR2RGB,
    )

    face = (
        face.astype(np.float32)
        / 255.0
    )

    face = (
        face - 0.5
    ) / 0.5

    blob = cv2.dnn.blobFromImage(
        face
    )

    net.setInput(
        blob,
        "data",
    )

    try:
        output = net.forward(
            ["label"]
        )[0]

    except cv2.error:
        output = net.forward()

    values = np.asarray(
        output
    ).reshape(-1)

    if values.size == 1:
        index = int(
            round(
                float(values[0])
            )
        )

        confidence = None

    else:
        shifted = (
            values.astype(np.float32)
            - np.max(values)
        )

        probabilities = np.exp(
            shifted
        )

        probabilities /= max(
            float(
                np.sum(probabilities)
            ),
            1e-9,
        )

        index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[index]
        )

    if not 0 <= index < len(
        EMOTIONS
    ):
        return "NEUTRAL", confidence

    return (
        EMOTIONS[index],
        confidence,
    )


def emotion_to_state(emotion):
    if emotion in (
        "ANGRY",
        "DISGUST",
    ):
        return "TENSE"

    if emotion == "SAD":
        return "STRESSED"

    return "NORMAL"


# ---------------------------------------------------------
# EYE / FATIGUE
# ---------------------------------------------------------

def eyes_are_open(
    aligned_face,
    eye_detector,
):
    if (
        aligned_face is None
        or aligned_face.size == 0
    ):
        return None

    gray = cv2.cvtColor(
        aligned_face,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.equalizeHist(
        gray
    )

    # Eyes are searched only in the upper area.
    upper = gray[
        8:68,
        8:104,
    ]

    eyes = eye_detector.detectMultiScale(
        upper,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(14, 14),
    )

    return len(eyes) >= 1


# ---------------------------------------------------------
# NUCLEO
# ---------------------------------------------------------

def connect_nucleo():
    if NucleoController is None:
        log(
            "nucleo_controller.py not found."
        )
        return None

    try:
        nucleo = NucleoController()

        if nucleo.connect() is False:
            log(
                "Nucleo connection failed."
            )
            return None

        log("Nucleo connected.")
        return nucleo

    except Exception as error:
        log(
            f"Nucleo connection error: {error}"
        )
        return None


def send_state(
    nucleo,
    state,
):
    if nucleo is None:
        return False

    try:
        nucleo.send_command(
            state
        )

        log(
            f"Nucleo command sent: {state}"
        )

        return True

    except Exception as error:
        log(
            f"Nucleo command error: {error}"
        )

        return False


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

def draw_panel(
    frame,
    connected,
    emotion,
    state,
    eye_text,
    fps,
):
    overlay = frame.copy()

    panel_width = min(
        360,
        frame.shape[1] - 20,
    )

    panel_height = min(
        185,
        frame.shape[0] - 20,
    )

    cv2.rectangle(
        overlay,
        (10, 10),
        (
            10 + panel_width,
            10 + panel_height,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.68,
        frame,
        0.32,
        0,
        frame,
    )

    rows = [
        (
            "Nucleo  : "
            + (
                "CONNECTED"
                if connected
                else "DISCONNECTED"
            )
        ),
        f"Emotion : {emotion}",
        f"State   : {state}",
        f"Eyes    : {eye_text}",
        f"FPS     : {fps:.1f}",
    ]

    y = 40

    for row in rows:
        cv2.putText(
            frame,
            row,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 32


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    log(
        "RPi-MASS optimized YuNet version starting."
    )

    show_splash()

    (
        fer_net,
        yunet,
        eye_detector,
    ) = load_models()

    camera = open_camera()
    nucleo = connect_nucleo()

    open_fullscreen_window()

    # Separate background workers prevent the camera loop from waiting
    # for YuNet or FER inference.
    face_executor = ThreadPoolExecutor(
        max_workers=1
    )

    emotion_executor = ThreadPoolExecutor(
        max_workers=1
    )

    face_future = None
    emotion_future = None

    frame_number = 0

    fps = 0.0
    fps_count = 0
    fps_time = time.time()

    last_face = None
    last_aligned_face = None

    shown_emotion = "WAITING"
    shown_state = "WAITING"
    eye_text = "WAITING"

    candidate_state = ""
    candidate_count = 0

    current_state = ""
    last_sent_state = ""
    last_send_time = 0.0

    eye_history = deque(maxlen=EYE_HISTORY_SIZE)
    fatigue_candidate_since = None
    fatigue_recovery_since = None
    fatigue_latched = False

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                raise RuntimeError(
                    "Camera frame could not be read."
                )

            frame = cv2.flip(
                frame,
                1,
            )

            frame_number += 1
            fps_count += 1

            now = time.time()

            if (
                now - fps_time
                >= 1.0
            ):
                fps = (
                    fps_count
                    / (now - fps_time)
                )

                fps_count = 0
                fps_time = now

            # ---------------------------------------------
            # Start YuNet in background
            # ---------------------------------------------

            if (
                frame_number
                % FACE_DETECT_EVERY
                == 0
                and face_future is None
            ):
                face_future = (
                    face_executor.submit(
                        detect_face_yunet,
                        yunet,
                        frame.copy(),
                    )
                )

            # ---------------------------------------------
            # Receive YuNet result
            # ---------------------------------------------

            if (
                face_future is not None
                and face_future.done()
            ):
                try:
                    last_face = (
                        face_future.result()
                    )

                    if last_face is not None:
                        last_aligned_face = (
                            align_face(
                                frame,
                                last_face,
                            )
                        )
                    else:
                        last_aligned_face = None

                except Exception as error:
                    log(
                        f"YuNet error: {error}"
                    )

                    last_face = None
                    last_aligned_face = None

                face_future = None

            # ---------------------------------------------
            # Draw face
            # ---------------------------------------------

            if last_face is not None:
                x, y, w, h = [
                    int(value)
                    for value
                    in last_face[:4]
                ]

                x = max(0, x)
                y = max(0, y)

                w = max(
                    1,
                    min(
                        w,
                        frame.shape[1] - x,
                    ),
                )

                h = max(
                    1,
                    min(
                        h,
                        frame.shape[0] - y,
                    ),
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2,
                )

            # ---------------------------------------------
            # Eye closure / fatigue
            # Rolling-window logic prevents one missed eye
            # detection from immediately causing FATIGUED.
            # ---------------------------------------------

            if (
                last_aligned_face is not None
                and frame_number
                % EYE_CHECK_EVERY
                == 0
            ):
                eye_result = eyes_are_open(
                    last_aligned_face,
                    eye_detector,
                )

                if eye_result is not None:
                    eye_history.append(
                        bool(eye_result)
                    )

            if len(eye_history) >= 4:
                open_ratio = (
                    sum(1 for value in eye_history if value)
                    / len(eye_history)
                )

                if open_ratio >= FATIGUE_EXIT_OPEN_RATIO:
                    eye_text = "OPEN"
                elif open_ratio <= FATIGUE_ENTER_OPEN_RATIO:
                    eye_text = "CLOSED"
                else:
                    eye_text = "PARTIAL"

                # Enter fatigue only after a sustained mostly-closed window.
                if not fatigue_latched:
                    if (
                        len(eye_history) == EYE_HISTORY_SIZE
                        and open_ratio <= FATIGUE_ENTER_OPEN_RATIO
                    ):
                        if fatigue_candidate_since is None:
                            fatigue_candidate_since = now

                        closed_duration = (
                            now - fatigue_candidate_since
                        )

                        eye_text = f"CLOSED {closed_duration:.1f}s"

                        if closed_duration >= FATIGUE_SECONDS:
                            fatigue_latched = True
                            fatigue_candidate_since = None
                            fatigue_recovery_since = None
                    else:
                        fatigue_candidate_since = None

                # Once fatigued, require sustained mostly-open eyes
                # before returning to the emotion-derived state.
                else:
                    if open_ratio >= FATIGUE_EXIT_OPEN_RATIO:
                        if fatigue_recovery_since is None:
                            fatigue_recovery_since = now

                        recovery_duration = (
                            now - fatigue_recovery_since
                        )

                        eye_text = f"OPEN {recovery_duration:.1f}s"

                        if recovery_duration >= FATIGUE_RECOVERY_SECONDS:
                            fatigue_latched = False
                            fatigue_recovery_since = None
                            fatigue_candidate_since = None

                            # Force the next state to be resent to Nucleo.
                            last_sent_state = ""
                    else:
                        fatigue_recovery_since = None

            else:
                eye_text = "CALIBRATING"

            fatigue_active = fatigue_latched

            # ---------------------------------------------
            # Start FER in background
            # ---------------------------------------------

            if (
                last_aligned_face is not None
                and emotion_future is None
                and frame_number
                % EMOTION_EVERY
                == 0
            ):
                emotion_future = (
                    emotion_executor.submit(
                        infer_emotion,
                        fer_net,
                        last_aligned_face.copy(),
                    )
                )

            # ---------------------------------------------
            # Receive FER result
            # ---------------------------------------------

            if (
                emotion_future is not None
                and emotion_future.done()
            ):
                try:
                    emotion, _ = (
                        emotion_future.result()
                    )

                    shown_emotion = emotion

                    raw_state = (
                        emotion_to_state(
                            emotion
                        )
                    )

                    if (
                        raw_state
                        == candidate_state
                    ):
                        candidate_count += 1

                    else:
                        candidate_state = (
                            raw_state
                        )

                        candidate_count = 1

                    if (
                        candidate_count
                        >= STATE_CONFIRM_COUNT
                    ):
                        current_state = (
                            raw_state
                        )

                except Exception as error:
                    log(
                        f"Emotion inference error: {error}"
                    )

                emotion_future = None

            # ---------------------------------------------
            # State priority
            # ---------------------------------------------

            if last_face is None:
                shown_emotion = (
                    "FACE NOT DETECTED"
                )

                shown_state = "WAITING"

                candidate_state = ""
                candidate_count = 0

                eye_history.clear()
                fatigue_candidate_since = None
                fatigue_recovery_since = None
                fatigue_latched = False
                eye_text = "NO FACE"

            else:
                if fatigue_active:
                    shown_state = (
                        "FATIGUED"
                    )

                elif current_state:
                    shown_state = (
                        current_state
                    )

                else:
                    shown_state = (
                        "WAITING"
                    )

            # ---------------------------------------------
            # Nucleo output
            # ---------------------------------------------

            desired_state = None

            if (
                last_face is not None
                and shown_state
                in (
                    "NORMAL",
                    "TENSE",
                    "STRESSED",
                    "FATIGUED",
                )
            ):
                desired_state = shown_state

            if desired_state is not None:
                state_changed = (
                    desired_state
                    != last_sent_state
                )

                refresh_due = (
                    now
                    - last_send_time
                    >= COMMAND_REFRESH_INTERVAL
                )

                interval_ok = (
                    now
                    - last_send_time
                    >= COMMAND_INTERVAL
                )

                if (
                    interval_ok
                    and (
                        state_changed
                        or refresh_due
                    )
                ):
                    if send_state(
                        nucleo,
                        desired_state,
                    ):
                        last_sent_state = (
                            desired_state
                        )

                        last_send_time = now

            # ---------------------------------------------
            # Display
            # ---------------------------------------------

            display = resize_to_fill(
                frame,
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
            )

            draw_panel(
                display,
                nucleo is not None,
                shown_emotion,
                shown_state,
                eye_text,
                fps,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                27,
                ord("q"),
            ):
                break

    finally:
        if (
            face_future is not None
            and not face_future.done()
        ):
            face_future.cancel()

        if (
            emotion_future is not None
            and not emotion_future.done()
        ):
            emotion_future.cancel()

        face_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        emotion_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        try:
            if nucleo is not None:
                nucleo.close()
        except Exception:
            pass

        camera.release()
        cv2.destroyAllWindows()

        log(
            "RPi-MASS optimized YuNet version stopped."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        error = traceback.format_exc()

        log(error)
        print(error)

        sys.exit(1)
