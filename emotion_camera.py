import time
from pathlib import Path

import cv2
import numpy as np

try:
    from arduino_controller import ArduinoController
except ImportError:
    ArduinoController = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion-ferplus-8.onnx"
FACE_CASCADE_PATH = Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

CAMERA_INDEX = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
PROCESS_EVERY_N_FRAMES = 5
STABLE_FRAME_COUNT = 2
COMMAND_INTERVAL = 1.0
MIN_CONFIDENCE = 0.25

EMOTION_LABELS = [
    "NEUTRAL", "HAPPY", "SURPRISE", "SAD",
    "ANGRY", "DISGUST", "FEAR", "CONTEMPT",
]


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    values -= np.max(values)
    exp_values = np.exp(values)
    total = np.sum(exp_values)
    return exp_values / total if total > 0 else np.zeros_like(values)


def emotion_to_state(emotion):
    if emotion in ("ANGRY", "DISGUST", "FEAR", "CONTEMPT"):
        return "GERGIN"
    if emotion == "SAD":
        return "STRESLI"
    return "NORMAL"


def connect_arduino():
    if ArduinoController is None:
        print("ArduinoController bulunamadi. Kamera modu devam ediyor.")
        return None
    try:
        arduino = ArduinoController()
        connected = arduino.connect()
        if connected is False:
            print("Arduino baglantisi kurulamadi. Kamera modu devam ediyor.")
            return None
        print("Arduino baglantisi basarili.")
        return arduino
    except Exception as error:
        print(f"Arduino baglanti hatasi: {error}")
        return None


def send_state_to_arduino(arduino, state):
    if arduino is None:
        print(f"Arduino bagli degil. Algilanan durum: {state}")
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
        print(f"Arduino komutu gonderildi: {state}")
    except Exception as error:
        print(f"Arduino komutu gonderilemedi: {error}")


def close_arduino(arduino):
    if arduino is None:
        return
    try:
        close_method = getattr(arduino, "close", None)
        if callable(close_method):
            close_method()
    except Exception as error:
        print(f"Arduino kapatma hatasi: {error}")


def load_emotion_model():
    if not MODEL_PATH.exists():
        print(f"HATA: ONNX modeli bulunamadi: {MODEL_PATH}")
        return None
    try:
        model = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        print("Duygu modeli basariyla yuklendi.")
        return model
    except cv2.error as error:
        print("HATA: Duygu modeli yuklenemedi.")
        print(error)
        return None


def load_face_detector():
    if not FACE_CASCADE_PATH.exists():
        print(f"HATA: Haar Cascade bulunamadi: {FACE_CASCADE_PATH}")
        return None
    detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if detector.empty():
        print("HATA: Yuz algilama modeli acilamadi.")
        return None
    print("Yuz algilama modeli basariyla yuklendi.")
    return detector


def open_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("HATA: Kamera acilamadi.")
        return None
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Kamera basariyla acildi.")
    return camera


def detect_emotion(emotion_model, face_gray):
    resized_face = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(
        resized_face,
        scalefactor=1.0,
        size=(64, 64),
        mean=(0,),
        swapRB=False,
        crop=False,
    )
    emotion_model.setInput(blob)
    scores = emotion_model.forward().flatten()
    probabilities = softmax(scores)
    emotion_index = int(np.argmax(probabilities))
    confidence = float(probabilities[emotion_index])
    return EMOTION_LABELS[emotion_index], confidence


def main():
    print("RPi-MASS duygu algilama sistemi baslatiliyor...")

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

    frame_count = 0
    last_faces = []
    candidate_state = ""
    candidate_count = 0
    last_sent_state = ""
    last_command_time = 0.0
    shown_emotion = "BEKLEME"
    shown_state = "BEKLEME"
    confidence = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Kameradan goruntu alinamadi.")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                small_gray = cv2.resize(
                    gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                )

                detected_faces = face_detector.detectMultiScale(
                    small_gray,
                    scaleFactor=1.2,
                    minNeighbors=4,
                    minSize=(40, 40),
                )

                last_faces = [
                    (x * 2, y * 2, w * 2, h * 2)
                    for x, y, w, h in detected_faces
                ]

                if last_faces:
                    x, y, w, h = max(last_faces, key=lambda f: f[2] * f[3])
                    face_gray = gray[y:y + h, x:x + w]

                    if face_gray.size > 0:
                        emotion, confidence = detect_emotion(emotion_model, face_gray)
                        shown_emotion = emotion
                        state = "NORMAL" if confidence < MIN_CONFIDENCE else emotion_to_state(emotion)
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
                            and now - last_command_time >= COMMAND_INTERVAL
                        ):
                            send_state_to_arduino(arduino, state)
                            last_sent_state = state
                            last_command_time = now
                else:
                    shown_emotion = "YUZ YOK"
                    shown_state = "BEKLEME"
                    confidence = 0.0
                    candidate_state = ""
                    candidate_count = 0

            for x, y, w, h in last_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            cv2.putText(frame, f"Duygu: {shown_emotion}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Durum: {shown_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Guven: %{confidence * 100:.0f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("RPi-MASS Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Program durduruldu.")
    finally:
        close_arduino(arduino)
        camera.release()
        cv2.destroyAllWindows()
        print("Program sonlandi.")


if __name__ == "__main__":
    main()
