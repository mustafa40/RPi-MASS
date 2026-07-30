import time
from pathlib import Path

import cv2
import numpy as np

from arduino_controller import ArduinoController


# =========================================================
# DOSYA VE KAMERA AYARLARI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emotion-ferplus-8.onnx"

FACE_CASCADE_PATH = (
    "/usr/share/opencv4/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

CAMERA_INDEX = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

MIN_CONFIDENCE = 0.25
STABLE_FRAME_COUNT = 3
COMMAND_INTERVAL = 1.0


# FER+ modelinin çıkış sırası
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
# YARDIMCI FONKSİYONLAR
# =========================================================

def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)

    exp_values = np.exp(values)
    total = np.sum(exp_values)

    if total <= 0:
        return np.zeros_like(values)

    return exp_values / total


def emotion_to_state(emotion):
    """
    FER+ duygu sınıfını araç içi sistem durumuna dönüştürür.
    """

    if emotion in (
        "ANGRY",
        "DISGUST",
        "FEAR",
        "CONTEMPT",
    ):
        return "GERGIN"

    if emotion == "SAD":
        return "STRESLI"

    return "NORMAL"


def connect_arduino():
    """
    Arduino bağlantısı kurulamazsa kamera sistemi
    tek başına çalışmaya devam eder.
    """

    try:
        arduino = ArduinoController()
        connected = arduino.connect()

        if connected is False:
            print("Arduino bağlantısı kurulamadı.")
            print("Sistem yalnızca kamera modunda çalışacak.")
            return None

        print("Arduino bağlantısı başarılı.")
        return arduino

    except Exception as error:
        print(f"Arduino bağlantı hatası: {error}")
        print("Sistem yalnızca kamera modunda çalışacak.")
        return None


def send_state_to_arduino(arduino, state):
    if arduino is None:
        print(f"Arduino bağlı değil. Algılanan durum: {state}")
        return

    try:
        if state == "GERGIN":
            arduino.tense()

        elif state == "STRESLI":
            arduino.stressed()

        elif state == "YORGUN":
            arduino.fatigue()

        else:
            arduino.normal()

        print(f"Arduino komutu gönderildi: {state}")

    except Exception as error:
        print(f"Arduino komutu gönderilemedi: {error}")


def close_arduino(arduino):
    if arduino is None:
        return

    try:
        close_method = getattr(arduino, "close", None)

        if callable(close_method):
            close_method()

    except Exception as error:
        print(f"Arduino kapatma hatası: {error}")


def load_emotion_model():
    if not MODEL_PATH.exists():
        print("HATA: ONNX modeli bulunamadı.")
        print(f"Aranan dosya: {MODEL_PATH}")
        return None

    try:
        model = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        print("Duygu modeli başarıyla yüklendi.")
        return model

    except cv2.error as error:
        print("HATA: Duygu modeli yüklenemedi.")
        print(error)
        return None


def load_face_detector():
    cascade_path = Path(FACE_CASCADE_PATH)

    if not cascade_path.exists():
        print("HATA: Haar Cascade dosyası bulunamadı.")
        print(f"Aranan dosya: {cascade_path}")
        return None

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        print("HATA: Yüz algılama modeli açılamadı.")
        return None

    print("Yüz algılama modeli başarıyla yüklendi.")
    return detector


def open_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("HATA: Kamera açılamadı.")
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Kamera başarıyla açıldı.")
    return camera


def detect_emotion(emotion_model, face_gray):
    """
    Yüz görüntüsünü FER+ modeline gönderir.
    Duygu adı ve güven oranını döndürür.
    """

    resized_face = cv2.resize(
        face_gray,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    blob = cv2.dnn.blobFromImage(
        resized_face,
        scalefactor=1.0,
        size=(64, 64),
        mean=(0,),
        swapRB=False,
        crop=False,
    )

    emotion_model.setInput(blob)

    output = emotion_model.forward()
    scores = output.flatten()

    probabilities = softmax(scores)

    emotion_index = int(np.argmax(probabilities))
    confidence = float(probabilities[emotion_index])

    emotion = EMOTION_LABELS[emotion_index]

    return emotion, confidence


# =========================================================
# ANA PROGRAM
# =========================================================

def main():
    print("RPi-MASS duygu algılama sistemi başlatılıyor...")

    emotion_model = load_emotion_model()

    if emotion_model is None:
        return

    face_detector = load_face_detector()

    if face_detector is None:
        return

    camera = open_camera()

    if camera is None:
        return

    arduino = connect_arduino()

    candidate_state = ""
    candidate_count = 0

    last_sent_state = ""
    last_command_time = 0.0
    try:
        
        while True:

            success, frame = camera.read()

            if not success:
                print("Kameradan görüntü alınamadı.")
                break

            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(70, 70)
            )

            shown_emotion = "YUZ YOK"
            shown_state = "BEKLEME"
            confidence = 0.0

            if len(faces) > 0:

                x, y, w, h = max(
                    faces,
                    key=lambda f: f[2] * f[3]
                )

                face = gray[y:y+h, x:x+w]

                if face.size > 0:

                    emotion, confidence = detect_emotion(
                        emotion_model,
                        face
                    )

                    shown_emotion = emotion

                    if confidence < MIN_CONFIDENCE:
                        state = "NORMAL"
                    else:
                        state = emotion_to_state(emotion)

                    shown_state = state

                    if state == candidate_state:
                        candidate_count += 1
                    else:
                        candidate_state = state
                        candidate_count = 1

                    now = time.time()

                    if (
                        candidate_count >= STABLE_FRAME_COUNT
                        and state != last_sent_state
                        and now - last_command_time > COMMAND_INTERVAL
                    ):

                        send_state_to_arduino(
                            arduino,
                            state
                        )

                        last_sent_state = state
                        last_command_time = now

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 255, 255),
                        2
                    )

            else:
                candidate_state = ""
                candidate_count = 0

            cv2.putText(
                frame,
                f"Duygu : {shown_emotion}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Durum : {shown_state}",
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Guven : %{confidence*100:.0f}",
                (20,105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255,255,255),
                2
            )

            cv2.imshow(
                "RPi-MASS Emotion Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Program durduruldu.")

    finally:

        close_arduino(arduino)

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("Program sonlandi.")


if __name__ == "__main__":
    main()
