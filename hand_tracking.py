import cv2
import mediapipe as mp
import time
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/handtrack")
def Handtrack():

    cap=cv2.VideoCapture(0)

    mphands=mp.solutions.hands
    hands=mphands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)#these are default that are already defines so no need to define it like this unless we want change
    mpDraw=mp.solutions.drawing_utils
    ptime=0
    ctime=0
    cap.set(3,1080)
    cap.set(4,720)

    while True:
        rel,frame=cap.read()

        framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(framergb)
        
        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame,handlandmark,mphands.HAND_CONNECTIONS)

        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        cv2.putText(frame,str(int(fps)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1)==ord("x"):
            break
    cv2.destroyAllWindows()