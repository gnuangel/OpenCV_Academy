
import cv2

camera = True

if __name__ == "__main__":

    faceCascade = cv2.CascadeClassifier('./venv/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
    profileFaceCascade = cv2.CascadeClassifier('./venv/lib/python3.12/site-packages/cv2/data/haarcascade_profileface.xml')
    eyeCascade = cv2.CascadeClassifier('./venv/lib/python3.12/site-packages/cv2/data/haarcascade_eye.xml')


    frameWidth = 320
    frameHeight = 240
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10,150)


    while True:
        ret, frame = cap.read(0)
        if frame is not None:
            frame = cv2.resize(frame, (1024,768))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30)
            )
            profileFaces = profileFaceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30,30)
            )
            eyes = eyeCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=10,
                minSize=(20,20)
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            for (x, y, w, h) in profileFaces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break
cv2.destroyAllWindows()
