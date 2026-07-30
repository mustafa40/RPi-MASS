import time
from pathlib import Path

import cv2
import numpy as np

try:
    from arduino_controller import ArduinoController
except ImportError:
    ArduinoController = None


# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion-ferplus-8.onnx"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Duygunun Arduino'ya gönderilmeden önce kaç kare
# art arda görülmesi gerektiği
REQUIRED_STABLE_FRAMES = 8

# Arduino'ya iki komut arasında minimum süre
COMMAND_DELAY_SECONDS = 3.0

# Çok düşük güven değerlerinde NORMAL kabul edilir
MIN_CONFIDENCE = 0.25


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


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------

def softmax(values):
    values = values.astype(np.float32)
    values = values - np.max(values)

    exp_values = np.exp(values)
    total = np.sum(exp_values)

    if total == 0:
        return np.zeros_like(values)

    return exp_values / total


def emotion_to_vehicle_state(emotion):
    """
    Kamera duygusunu Arduino komutuna dönüştürür.
    """

    if emotion in ("ANGRY", "DISGUST", "FEAR", "CONTEMPT"):
        return "GERGIN"

    if emotion == "SAD":
        return "STRESLI"

    return "NORMAL"


def send_arduino_state(arduino, state):
    """
    ArduinoController içerisindeki uygun fonksiyonu çağırır.
    """

    if arduino is None:
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


def open_arduino():
    """
    Arduino bağlı değilse programın kamera kısmı yine çalışır.
    """

    if ArduinoController is None:
        print("ArduinoController bulunamadı.")
        print("Program yalnızca kamera modunda devam ediyor.")
        return None

    try:
        arduino = ArduinoController()

        connected = arduino.connect()

        if connected is False:
            print("Arduino bağlantısı kurulamadı.")
            print("Program yalnızca kamera modunda devam ediyor.")
            return None

        print("Arduino bağlantısı başarılı.")
        return arduino

    except Exception as error:
        print(f"Arduino bağlantı hatası: {error}")
        print("Program yalnızca kamera modunda devam ediyor.")
        return None


def close_arduino(arduino):
    if arduino is None:
        return

    try:
        # ArduinoController içinde close varsa çalıştır
        close_function = getattr(arduino, "close", None)

        if callable(close_function):
            close_function()

    except Exception as error:
        print(f"Arduino kapatma hatası: {error}")


# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------

def main():
    print("RPi-MASS duygu algılama sistemi başlatılıyor...")

    if not MODEL_PATH.exists():
        print("HATA: ONNX modeli bulunamadı.")
        print(f"Aranan dosya: {MODEL_PATH}")
        return

    # ONNX modelini yükle
    try:
        emotion_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        print("Duygu modeli başarıyla yüklendi.")

    except cv2.error as error:
        print("Duygu modeli yüklenemedi:")
        print(error)
        return

    # OpenCV içerisindeki hazır Haar yüz modeli
    face_cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(face_cascade_path)

    if face_detector.empty():
        print("HATA: Yüz algılama modeli yüklenemedi.")
        return

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("HATA: Kamera açılamadı.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    arduino = open_arduino()

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

            # Ayna görüntüsü
            frame = cv2.flip(frame, 1)

            gray_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(100, 100)
            )

            displayed_emotion = "YUZ YOK"
            displayed_state = "BEKLEME"
            confidence = 0.0

            if len(faces) > 0:
                # En büyük yüzü seç
                x, y, width, height = max(
                    faces,
                    key=lambda face: face[2] * face[3]
                )

                face_gray = gray_frame[
                    y:y + height,
                    x:x + width
                ]

                if face_gray.size > 0:
                    face_gray = cv2.resize(
                        face_gray,
                        (64, 64),
                        interpolation=cv2.INTER_AREA
                    )

                    # Model girişi: 1 x 1 x 64 x 64
                    blob = cv2.dnn.blobFromImage(
                        face_gray,
                        scalefactor=1.0,
                        size=(64, 64),
                        mean=(0,),
                        swapRB=False,
                        crop=False
                    )

                    emotion_net.setInput(blob)

                    output = emotion_net.forward()
                    scores = output.flatten()

                    probabilities = softmax(scores)

                    emotion_index = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[emotion_index]
                    )

                    displayed_emotion = EMOTION_LABELS[
                        emotion_index
                    ]

                    if confidence < MIN_CONFIDENCE:
                        displayed_state = "NORMAL"
                    else:
                        displayed_state = emotion_to_vehicle_state(
                            displayed_emotion
                        )

                    # Aynı durum art arda görülüyor mu?
                    if displayed_state == candidate_state:
                        candidate_count += 1
                    else:
                        candidate_state = displayed_state
                        candidate_count = 1

                    current_time = time.time()

                    # Kararlı hale geldiyse Arduino'ya gönder
                    if (
                        candidate_count >= REQUIRED_STABLE_FRAMES
                        and candidate_state != last_sent_state
                        and current_time - last_command_time
                        >= COMMAND_DELAY_SECONDS
                    ):
                        send_arduino_state(
                            arduino,
                            candidate_state
                        )

                        last_sent_state = candidate_state
                        last_command_time = current_time

                    # Yüz çerçevesi
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (255, 255, 255),
                        2
                    )

            else:
                candidate_state = ""
                candidate_count = 0

            # Ekran bilgileri
            cv2.putText(
                frame,
                f"Duygu: {displayed_emotion}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Durum: {displayed_state}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Guven: %{confidence * 100:.0f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Cikis: Q",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.imshow("RPi-MASS Emotion Detection", frame)

            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")

    finally:
        close_arduino(arduino)
        camera.release()
        cv2.destroyAllWindows()
        print("Program kapatıldı.")


if __name__ == "__main__":
    main()
