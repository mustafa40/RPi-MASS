import time
import cv2
import mediapipe as mp

from arduino_controller import ArduinoController
from tension_detector import TensionDetector


mp_face_mesh = mp.solutions.face_mesh


def distance(point_a, point_b):
    return (
        (point_a.x - point_b.x) ** 2 +
        (point_a.y - point_b.y) ** 2
    ) ** 0.5


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Kamera açılamadı.")
        return

    arduino = ArduinoController()
    arduino_connected = arduino.connect()

    detector = TensionDetector(
        threshold=0.62,
        required_frames=12
    )

    last_state = "NORMAL"
    last_command_time = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(rgb_frame)

            state = "YUZ_YOK"
            score = 0.0

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                face_width = distance(
                    landmarks[234],
                    landmarks[454]
                )

                if face_width > 0:
                    inner_brow_distance = distance(
                        landmarks[107],
                        landmarks[336]
                    ) / face_width

                    left_brow_eye = distance(
                        landmarks[105],
                        landmarks[159]
                    ) / face_width

                    right_brow_eye = distance(
                        landmarks[334],
                        landmarks[386]
                    ) / face_width

                    brow_lowering = (
                        left_brow_eye + right_brow_eye
                    ) / 2

                    lip_opening = distance(
                        landmarks[13],
                        landmarks[14]
                    ) / face_width

                    tension = detector.update(
                        brow_distance_ratio=inner_brow_distance,
                        brow_lowering_ratio=brow_lowering,
                        lip_compression_ratio=lip_opening
                    )

                    score = tension.score
                    state = "GERGIN" if tension.tense else "NORMAL"

                    now = time.time()

                    if (
                        state != last_state and
                        now - last_command_time > 2.0
                    ):
                        if arduino_connected:
                            if state == "GERGIN":
                                arduino.tense()
                            else:
                                arduino.normal()

                        last_state = state
                        last_command_time = now

            cv2.putText(
                frame,
                f"Durum: {state}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Gerginlik skoru: {score:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("RPi-MASS Gerginlik Testi", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if arduino_connected:
        arduino.turn_off()
        arduino.close()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
