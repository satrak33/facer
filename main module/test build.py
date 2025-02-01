import face_recognition as fr
import sys
import cv2
import numpy as np
import math
from datetime import datetime
from threading import Thread, Lock
from queue import Queue

# URL джерела відео
SOURCE = 'https://192.168.0.100:8080/video'


def face_confidence(face_distance, face_match=0.1):
    range_val = (1.0 - face_match)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match:
        return f"{round(linear_val * 100, 2)}%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return f"{round(value, 2)}%"


class FaceRecognition:
    def __init__(self):
        # Завантаження закодованих облич
        self.known_face_encodings = np.load('know_face_encodings.npy')
        self.known_face_names = np.load('know_face_names.npy')

        # Буфери для потоків
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)

        # Блокування для потоків
        self.lock = Lock()

        # Змінна для контролю завершення
        self.running = True

    def process_frame(self):
        """Обробка кадру в окремому потоці."""
        while self.running or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Зменшення розміру кадру
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Розпізнавання облич
                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = fr.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    else:
                        name = "Unknown"
                        confidence = "(Unknown)"

                    face_names.append(f"{name} ({confidence})")

                # Масштабування координат облич назад до початкового розміру
                scaled_face_locations = [
                    (top * 4, right * 4, bottom * 4, left * 4)
                    for (top, right, bottom, left) in face_locations
                ]

                # Додавання результатів у чергу
                self.result_queue.put((scaled_face_locations, face_names, frame))

    def run_recognition(self):
        """Запуск розпізнавання облич."""
        vid = cv2.VideoCapture(SOURCE)
        if not vid.isOpened():
            sys.exit('Video source not found...')

        print(f'Video FPS: {vid.get(cv2.CAP_PROP_FPS)}')

        # Запуск потоку для обробки кадрів
        processing_thread = Thread(target=self.process_frame, daemon=True)
        processing_thread.start()

        while True:
            ret, frame = vid.read()
            if int(vid.get(1)) % 25 != 0:
                continue
            # Додавання кадру в чергу
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            # Обробка результатів
            if not self.result_queue.empty():
                scaled_face_locations, face_names, display_frame = self.result_queue.get()

                # Відображення результатів
                for (top, right, bottom, left), name in zip(scaled_face_locations, face_names):
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(display_frame, (left, top - 35), (right, top), (0, 0, 255), -1)
                    cv2.putText(display_frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow('Face Recognition', display_frame)

            if cv2.waitKey(1) == ord('q'):
                self.running = False
                break

        # Завершення потоків
        processing_thread.join()
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recognition = FaceRecognition()
    face_recognition.run_recognition()
