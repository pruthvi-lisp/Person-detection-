import cv2

full_body = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('Pedestrian overpass.mp4')
while cap.isOpened():

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = full_body.detectMultiScale(gray, 1.2, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in body:
        cv2.putText(frame, 'Person', (x, y - 5), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow('Body Detection', frame)

    k = cv2.waitKey(1)
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()