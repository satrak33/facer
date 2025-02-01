import cv2
import time
import face_recognition as fr
import numpy as np

def face_detect(img, facedetect):

    start = time.time()
    # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_small_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    face_rec = []
    for (x, y, w, h) in faces:
        face_rec.append([y, x + w, y + h, x])
    print(face_rec)
    face_encoding = fr.face_encodings(rgb_small_frame, face_rec)
    print(face_encoding)

    face_recognitions = np.load('../main module/know_face_encodings.npy')
    print(fr.compare_faces(face_recognitions, face_encoding))
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)

    cv2.imshow('Frame', img)
    print(time.time() - start)


def main():
    for i in range(1):
        facedetect = cv2.CascadeClassifier('../assets/haarcascade_frontalface_default.xml')
        img = cv2.imread('exemple_for_test.jpg')
        face_detect(img, facedetect)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()