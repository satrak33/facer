import math

import cv2
import face_recognition as fr
import numpy as np

source = 'https://192.168.0.103:8080/video'


def face_confidence(face_distance: float, face_match: float = 0.6) -> str:
    range: float = (1.0 - face_match)
    linear_val: float = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations: list = []
    face_encodings: list = []
    face_names: list = []
    know_face_encodings: list = []
    know_face_names: list = []

    def __init__(self):
        self.load_faces()

    def load_faces(self):
        self.know_face_encodings = np.load('know_face_encodings.npy')
        self.know_face_names = np.load('know_face_names.npy')

    def run_recognition(self):
        vid = cv2.VideoCapture('0')
        vid.open(source)

        while True:
            ret, frame = vid.read()
            if int(vid.get(1)) % 50 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = fr.face_locations(rgb_small_frame)

                self.face_encodings = fr.face_encodings(rgb_small_frame, self.face_locations)

                if self.face_encodings:
                    self.face_names: list = []

                    for face_encoding in self.face_encodings:
                        matches: list = fr.compare_faces(self.know_face_encodings, face_encoding)
                        name: str = 'Unknown'
                        confidence: str = '(Unknown)'

                        face_distances = fr.face_distance(self.know_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.know_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        self.face_names.append(f'{name} ({confidence})')

                    for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), -1)
                        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('face', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    frec = FaceRecognition()
    frec.run_recognition()
