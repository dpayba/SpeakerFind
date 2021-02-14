import face_recognition
import cv2
from scipy.spatial import distance as dist
import math
import os

MOUTH_SPEAKING_SECONDS = 15

videoCapture = cv2.VideoCapture(0)

# Variable Declaration
count = 0
speakCount = 0
faceLocations = []
faceEncodings = []
faceNames = []

ret, frame = videoCapture.read(0)
smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
rgbFrame = smallFrame[:, :, ::-1]
faceLandmarks = face_recognition.face_landmarks(rgbFrame)

speaking = False
processFrame = True

while True:
    ret, frame = videoCapture.read()

    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgbFrame = smallFrame[:, :, ::-1]

    if processFrame:

        faceLandmarks = face_recognition.face_landmarks(rgbFrame)

        faceLocations = face_recognition.face_locations(rgbFrame)
        faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)

        for faceLandmark in faceLandmarks:
            p1 = faceLandmarks[0]['top_lip']
            p2 = faceLandmarks[0]['bottom_lip']
            x1, y1 = p1[9]
            x3, y3 = p1[8]
            x4, y4 = p1[10]
            x2, y2 = p2[9]
            x5, y5 = p2[8]
            x6, y6 = p2[10]
            dist = math.sqrt(((x2 + x5 + x6) - (x1 + x3 + x4)) ** 2 + ((y2 + y5 + y6) - (y1 + y3 + y4)) ** 2)
            #print(dist)

        open = dist > 5

        if (open):
            speakCount += 1
        else:
            speakCount = 0

        if (speakCount >= MOUTH_SPEAKING_SECONDS):
            speaking = True

        faceNames = []
        for encoding in faceEncodings:
            text = "Speaking Detected"
            faceNames.append(text)

    processFrame = not processFrame

    for (top, right, bottom, left), text in zip(faceLocations, faceNames):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        if (speaking):
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

            if count < 1:
                title = "SpeakerFind"
                message = "Dane is speaking"
                command = f'''
                                                    osascript -e 'display notification "{message}" with title "{title}"'
                                                    '''
                os.system(command)
                count += 1

            if cv2.waitKey(1) == 32:
                speaking = False
                count = 0
            speakCount = 0

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()



