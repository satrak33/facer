import face_recognition as fr
import cv2
import time
import numpy as np


def face_detect(img):

    start = time.time()
    # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    rgb_rame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = fr.face_locations(rgb_rame)
    print(faces)
    face_encoding = fr.face_encodings(rgb_rame, faces)
    face_recognitions = np.load('../know_face_encodings.npy')
    print(fr.compare_faces(face_recognitions, face_encoding))

    for (top, right, bottom, left) in faces:

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 10)

    cv2.imshow("Frame", img)

    print(time.time() - start)


def main():
    for i in range(1):
        img = cv2.imread('exemple_for_test.jpg')
        face_detect(img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()




