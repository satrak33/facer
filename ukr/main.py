import face_recognition as fr
import cv2
import numpy as np
from datetime import datetime

# Функція для обчислення впевненості в розпізнаванні обличчя
def face_confidence(face_distance, face_match=0.1):
    # Визначення діапазону значення впевненості
    range: float = (1.0 - face_match)
    linear_val: float = (1.0 - face_distance) / (range * 2.0)

    # Повернення впевненості у відсотках залежно від відстані обличчя
    if face_distance > face_match:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * pow((linear_val - 0.5) * 2, 0.2)) * 100)
        return str(round(value, 2)) + '%'

# Клас для реалізації функціоналу розпізнавання облич
class FaceRecognition:
    # Ініціалізація атрибутів для збереження даних про обличчя
    face_locations: list = []  # Координати розпізнаних облич
    face_encodings: list = []  # Кодовані дані облич
    face_names: list = []      # Імена розпізнаних облич
    know_face_encodings: list = []  # Кодовані дані відомих облич
    know_face_names: list = []      # Імена відомих облич

    # Конструктор класу: завантаження кодованих даних відомих облич
    def __init__(self):
        self.encode_faces()

    # Метод для завантаження даних з файлів
    def encode_faces(self):
        self.know_face_encodings = np.load('know_face_encodings.npy')
        self.know_face_names = np.load('know_face_names.npy')

    # Основний метод для запуску процесу розпізнавання облич
    def run_recognition(self):
        # Відкриття відеопотоку з веб-камери (індекс '0' означає першу камеру)
        vid = cv2.VideoCapture('0')

        if not vid.isOpened():
            raise "Відео не знайдено"

        while True:
            # Читання одного кадру з відеопотоку
            ret, frame = vid.read()

            # Виконання розпізнавання кожні 50 кадрів
            if int(vid.get(1)) % 50 == 0:
                # Зменшення розміру кадру для пришвидшення обробки
                small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
                # Конвертація кадру в RGB (face_recognition працює з RGB)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Знаходження облич та отримання їх кодованих даних
                self.face_locations = fr.face_locations(rgb_small_frame)
                self.face_encodings = fr.face_encodings(rgb_small_frame, self.face_locations)

                if self.face_encodings:
                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        # Порівняння обличчя з відомими кодованими даними
                        matches: list = fr.compare_faces(self.know_face_encodings, face_encoding)
                        name: str = 'НЕВІДОМИЙ'
                        confidence: str = '(НЕВІДОМИЙ)'

                        # Обчислення відстані до відомих облич
                        face_distances = fr.face_distance(self.know_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)  # Індекс найкращого співпадіння

                        if matches[best_match_index]:
                            name = self.know_face_names[best_match_index]  # Отримання імені
                            confidence = face_confidence(face_distances[best_match_index])  # Обчислення впевненості

                        self.face_names.append(f'{name} ({confidence})')

                    # Малювання рамок навколо розпізнаних облич та виведення імен
                    for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                        # Масштабування координат до розміру початкового кадру
                        top *= 10
                        right *= 10
                        bottom *= 10
                        left *= 10

                        # Встановлення кольору рамки: червоний для невідомих, зелений для відомих облич
                        color = (255, 0, 0) if name == 'НЕВІДОМИЙ' else (0, 255, 0)

                        # Малювання прямокутника навколо обличчя
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        # Додавання тексту з ім'ям та впевненістю
                        cv2.putText(frame, name, (left + 6, top-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

                # Відображення кадру з рамками та текстом
                cv2.imshow('face', frame)

            # Вихід з циклу при натисканні клавіші 'q'
            if cv2.waitKey(1) == ord('q'):
                break

        # Закриття відеопотоку та вікон OpenCV
        vid.release()
        cv2.destroyAllWindows()

# Головна функція програми
if __name__ == '__main__':
    frec = FaceRecognition()  # Створення об'єкта класу FaceRecognition
    frec.run_recognition()   # Запуск розпізнавання
