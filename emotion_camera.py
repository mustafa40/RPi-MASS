import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    from arduino_controller import ArduinoController
except ImportError:
    ArduinoController = None


# =========================================================
# DOSYA YOLLARI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emotion-ferplus-8.onnx"
LOGO_PATH = BASE_DIR / "mfi_logo.png"

FACE_CASCADE_PATH = Path(
    "/usr/share/opencv4/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# KAMERA VE UYGULAMA AYARLARI
# =========================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_NAME = "RPi-MASS Emotion Detection"
FULLSCREEN = True

# Yuz arama ve duygu analizi kac karede bir baslatilsin?
PROCESS_EVERY_N_FRAMES = 15

STABLE_FRAME_COUNT = 1
COMMAND_INTERVAL = 1.0
MIN_CONFIDENCE = 0.25


# =========================================================
# DUYGU ETIKETLERI
# =========================================================

EMOTION_LABELS = [
    "NEUTRAL",
    "HAPPY",
    "SURPRISE",
    "SAD",
    "ANGRY",
    "DISGUST",
    "FEAR",
    "CONTEMPT",
]


# =========================================================
# EKRANA YAZI YAZMA
# =========================================================

def draw_text(img, text, x, y, scale=0.62):
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )

    cv2.rectangle(
        img,
        (x - 5, y - text_height - 8),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# MFI ACILIS EKRANI
# =========================================================

def show_splash():
    logo = cv2.imread(str(LOGO_PATH))

    if logo is None:
        print(f"Logo okunamadi: {LOGO_PATH}")
        return

    splash_width = 1280
    splash_height = 720

    splash = np.zeros(
        (splash_height, splash_width, 3),
        dtype=np.uint8
    )

    logo_height, logo_width = logo.shape[:2]

    max_logo_width = 430
    max_logo_height = 320

    width_scale = max_logo_width / logo_width
    height_scale = max_logo_height / logo_height
    resize_scale = min(width_scale, height_scale)

    new_logo_width = max(
        1,
        int(logo_width * resize_scale)
    )

    new_logo_height = max(
        1,
        int(logo_height * resize_scale)
    )

    logo = cv2.resize(
        logo,
        (new_logo_width, new_logo_height),
        interpolation=cv2.INTER_AREA
    )

    logo_x = (
        splash_width - new_logo_width
    ) // 2

    logo_y = 70

    splash[
        logo_y:logo_y + new_logo_height,
        logo_x:logo_x + new_logo_width
    ] = logo

    first_title = "Connectivity & Smart Devices"
    second_title = "RPi-MASS"

    first_title_scale = 1.1
    second_title_scale = 1.7

    first_title_size, _ = cv2.getTextSize(
        first_title,
        cv2.FONT_HERSHEY_SIMPLEX,
        first_title_scale,
        2
    )

    second_title_size, _ = cv2.getTextSize(
        second_title,
        cv2.FONT_HERSHEY_SIMPLEX,
        second_title_scale,
        3
    )

    first_title_x = (
        splash_width - first_title_size[0]
    ) // 2

    second_title_x = (
        splash_width - second_title_size[0]
    ) // 2

    cv2.putText(
        splash,
        first_title,
        (first_title_x, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        first_title_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        splash,
        second_title,
        (second_title_x, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        second_title_scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    if FULLSCREEN:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    cv2.imshow(WINDOW_NAME, splash)
    cv2.waitKey(2500)
    cv2.destroyWindow(WINDOW_NAME)


# =========================================================
# YARDIMCI FONKSIYONLAR
# =========================================================

def softmax(values):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = values - np.max(values)

    exp_values = np.exp(values)
    total = np.sum(exp_values)

    if total <= 0:
        return np.zeros_like(values)

    return exp_values / total


def emotion_to_state(emotion):
    if emotion in (
        "ANGRY",
        "DISGUST",
        "FEAR",
        "CONTEMPT"
    ):
        return "GERGIN"

    if emotion == "SAD":
        return "STRESLI"

    return "NORMAL"


# =========================================================
# ARDUINO BAGLANTISI
# =========================================================

def connect_arduino():
    if ArduinoController is None:
        print(
            "ArduinoController bulunamadi. "
            "Kamera modu devam ediyor."
        )
        return None

    try:
        arduino = ArduinoController()
        connected = arduino.connect()

        if connected is False:
            print(
                "Arduino baglantisi kurulamadi. "
                "Kamera modu devam ediyor."
            )
            return None

        print("Arduino baglantisi basarili.")
        return arduino

    except Exception as error:
        print(
            f"Arduino baglanti hatasi: {error}"
        )
        return None


def send_state_to_arduino(arduino, state):
    if arduino is None:
        print(
            "Arduino bagli degil. "
            f"Algilanan durum: {state}"
        )
        return False

    try:
        if state == "GERGIN":
            arduino.tense()

        elif state == "STRESLI":
            arduino.stressed()

        elif state == "YORGUN":
            arduino.fatigue()

        else:
            arduino.normal()

        print(
            f"Arduino komutu gonderildi: {state}"
        )

        return True

    except Exception as error:
        print(
            "Arduino komutu gonderilemedi: "
            f"{error}"
        )
        return False


def close_arduino(arduino):
    if arduino is None:
        return

    try:
        close_method = getattr(
            arduino,
            "close",
            None
        )

        if callable(close_method):
            close_method()

    except Exception as error:
        print(
            f"Arduino kapatma hatasi: {error}"
        )


# =========================================================
# MODELLERI YUKLEME
# =========================================================

def load_emotion_model():
    if not MODEL_PATH.exists():
        print(
            "HATA: ONNX modeli bulunamadi: "
            f"{MODEL_PATH}"
        )
        return None

    try:
        model = cv2.dnn.readNetFromONNX(
            str(MODEL_PATH)
        )

        print(
            "Duygu modeli basariyla yuklendi."
        )

        return model

    except cv2.error as error:
        print(
            "HATA: Duygu modeli yuklenemedi."
        )
        print(error)
        return None


def load_face_detector():
    if not FACE_CASCADE_PATH.exists():
        print(
            "HATA: Haar Cascade bulunamadi: "
            f"{FACE_CASCADE_PATH}"
        )
        return None

    detector = cv2.CascadeClassifier(
        str(FACE_CASCADE_PATH)
    )

    if detector.empty():
        print(
            "HATA: Yuz algilama modeli "
            "acilamadi."
        )
        return None

    print(
        "Yuz algilama modeli basariyla "
        "yuklendi."
    )

    return detector


# =========================================================
# KAMERAYI ACMA
# =========================================================

def open_camera():
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if not camera.isOpened():
        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():
        print("HATA: Kamera acilamadi.")
        return None

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print("Kamera basariyla acildi.")
    return camera


# =========================================================
# DUYGU ANALIZI
# =========================================================

def detect_emotion(model, face_gray):
    try:
        if (
            face_gray is None
            or face_gray.size == 0
        ):
            return "NEUTRAL", 0.0

        resized_face = cv2.resize(
            face_gray,
            (64, 64),
            interpolation=cv2.INTER_AREA
        )

        normalized_face = (
            resized_face.astype(np.float32)
            / 255.0
        )

        blob = normalized_face.reshape(
            1,
            1,
            64,
            64
        )

        model.setInput(blob)
        prediction = model.forward()

        prediction = np.asarray(
            prediction,
            dtype=np.float32
        ).reshape(-1)

        probabilities = softmax(
            prediction
        )

        emotion_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[emotion_index]
        )

        if emotion_index >= len(
            EMOTION_LABELS
        ):
            return "NEUTRAL", 0.0

        emotion = EMOTION_LABELS[
            emotion_index
        ]

        return emotion, confidence

    except Exception as error:
        print(
            f"Duygu analiz hatasi: {error}"
        )
        return "NEUTRAL", 0.0
# =========================================================
# ANA PROGRAM
# =========================================================

def main():
    print(
        "RPi-MASS duygu algilama sistemi "
        "baslatiliyor..."
    )

    show_splash()

    emotion_model = load_emotion_model()

    if emotion_model is None:
        return

    face_detector = load_face_detector()

    if face_detector is None:
        return

    camera = open_camera()

    if camera is None:
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    if FULLSCREEN:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    arduino = connect_arduino()

    frame_count = 0
    last_faces = []

    candidate_state = ""
    candidate_count = 0

    last_sent_state = ""
    last_command_time = 0.0

    shown_emotion = "BEKLEME"
    shown_state = "BEKLEME"
    shown_command = "HENUZ GONDERILMEDI"

    confidence = 0.0

    fps = 0.0
    fps_counter = 0
    fps_start_time = time.time()

    # ONNX duygu analizi burada ayri
    # bir is parcaciginda calisacak.
    analysis_executor = ThreadPoolExecutor(
        max_workers=1
    )

    analysis_future = None

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Kameradan goruntu alinamadi."
                )
                break

            frame_count += 1
            fps_counter += 1

            fps_elapsed = (
                time.time() - fps_start_time
            )

            if fps_elapsed >= 1.0:
                fps = (
                    fps_counter / fps_elapsed
                )

                fps_counter = 0
                fps_start_time = time.time()

            frame = cv2.flip(
                frame,
                1
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Arka plandaki ONNX analizi
            # bittiyse sonucu burada al.
            if (
                analysis_future is not None
                and analysis_future.done()
            ):
                try:
                    emotion, confidence = (
                        analysis_future.result()
                    )

                    shown_emotion = emotion

                    if (
                        confidence
                        < MIN_CONFIDENCE
                    ):
                        state = "NORMAL"
                    else:
                        state = emotion_to_state(
                            emotion
                        )

                    shown_state = state

                    if state == candidate_state:
                        candidate_count += 1
                    else:
                        candidate_state = state
                        candidate_count = 1

                    now = time.time()

                    stable_enough = (
                        candidate_count
                        >= STABLE_FRAME_COUNT
                    )

                    state_changed = (
                        state
                        != last_sent_state
                    )

                    interval_passed = (
                        now - last_command_time
                        >= COMMAND_INTERVAL
                    )

                    if (
                        stable_enough
                        and state_changed
                        and interval_passed
                    ):
                        command_sent = (
                            send_state_to_arduino(
                                arduino,
                                state
                            )
                        )

                        shown_command = state

                        if command_sent:
                            last_sent_state = state
                            last_command_time = now

                except Exception as error:
                    print(
                        "Duygu analiz sonucu "
                        "okunamadi: "
                        f"{error}"
                    )

                analysis_future = None

            # Belirlenen kare araliginda
            # yuz tespiti yap.
            if (
                frame_count
                % PROCESS_EVERY_N_FRAMES
                == 0
            ):
                small_gray = cv2.resize(
                    gray,
                    None,
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA
                )

                detected_faces = (
                    face_detector.detectMultiScale(
                        small_gray,
                        scaleFactor=1.2,
                        minNeighbors=4,
                        minSize=(40, 40)
                    )
                )

                last_faces = [
                    (
                        int(x * 2),
                        int(y * 2),
                        int(w * 2),
                        int(h * 2)
                    )
                    for x, y, w, h
                    in detected_faces
                ]

                if last_faces:
                    largest_face = max(
                        last_faces,
                        key=lambda face:
                        face[2] * face[3]
                    )

                    x, y, w, h = largest_face

                    frame_height, frame_width = (
                        gray.shape[:2]
                    )

                    x = max(0, x)
                    y = max(0, y)

                    x_end = min(
                        frame_width,
                        x + w
                    )

                    y_end = min(
                        frame_height,
                        y + h
                    )

                    face_gray = gray[
                        y:y_end,
                        x:x_end
                    ]

                    # Yeni analiz sadece onceki
                    # analiz bittiyse baslatilir.
                    if (
                        face_gray.size > 0
                        and analysis_future is None
                    ):
                        analysis_future = (
                            analysis_executor.submit(
                                detect_emotion,
                                emotion_model,
                                face_gray.copy()
                            )
                        )

                else:
                    shown_emotion = "YUZ YOK"
                    shown_state = "BEKLEME"
                    confidence = 0.0

                    candidate_state = ""
                    candidate_count = 0

            # Yuz cercevelerini ekrana ciz.
            for x, y, w, h in last_faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2
                )

            if arduino is not None:
                arduino_status = "BAGLI"
            else:
                arduino_status = (
                    "BAGLI DEGIL"
                )

            draw_text(
                frame,
                f"Arduino : {arduino_status}",
                15,
                35
            )

            draw_text(
                frame,
                f"Duygu   : {shown_emotion}",
                15,
                67
            )

            draw_text(
                frame,
                f"Durum   : {shown_state}",
                15,
                99
            )

            draw_text(
                frame,
                f"Komut   : {shown_command}",
                15,
                131
            )

            draw_text(
                frame,
                (
                    "Guven   : "
                    f"%{confidence * 100:.0f}"
                ),
                15,
                163
            )

            draw_text(
                frame,
                f"FPS     : {fps:.1f}",
                15,
                195
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            pressed_key = (
                cv2.waitKey(1) & 0xFF
            )

            if pressed_key == ord("q"):
                break

            if pressed_key == 27:
                break

    except KeyboardInterrupt:
        print("Program durduruldu.")

    finally:
        if (
            analysis_future is not None
            and not analysis_future.done()
        ):
            analysis_future.cancel()

        analysis_executor.shutdown(
            wait=False,
            cancel_futures=True
        )

        close_arduino(arduino)

        camera.release()
        cv2.destroyAllWindows()

        print("Program sonlandi.")


if __name__ == "__main__":
    main()
