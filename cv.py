import cv2

scale0 = 0.33
scale1 = 3


def main():
    video = cv2.VideoCapture(0)
    video.open('https://192.168.0.102:8080/video')

    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray, (0, 0), fx=scale0, fy=scale0)

        faces = facedetect.detectMultiScale(small_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            x *= scale1
            y *= scale1
            w *= scale1
            h *= scale1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)

        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', 0, 0)

        k = cv2.waitKey(1)

        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()