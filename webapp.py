import cv2
# import urllib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("FACE-DETECT.h5")

# url = "http://192.168.146.11:8080/shot.jpg"
st.title("License Holder Recognition")
st.write("This is a simple application to recognize the license holder of a vehicle")
img = st.camera_input("Take a picture")
# if img is not None:
#     bytes_data = img.getvalue()
#     st.write(bytes_data)
# FRAME_WINDOW = st.image([])

def get_pred_label(pred):
    labels = ['Arnab', 'Ashutosh', 'Bijay-ID-1000', 'Durga', 'Malay', 'sambit']
    return labels[pred]

# def capture_face(video_capture):
#     for i in range(3):
#         video_capture.read()

#     while(True):
#         ret, frame = video_capture.read()
#         FRAME_WINDOW.image(frame[:,:,::-1])
#         small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
#         gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
#         faces = classifier.detectMultiScale(gray, 1.3, 5)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]
#             break
#         if len(faces) > 0:
#             break
#     return roi_gray, roi_color
#     if ret == False:
#         return ret, frame

ret = True
while ret:
    # img_url = urllib.request.urlopen(url)
    if img is not None:
        image = np.array(bytearray(img.read()),np.uint8)
        frame = cv2.imdecode(image,-1)
    
        faces = classifier.detectMultiScale(frame,1.5,5)
    
        for x,y,w,h in faces:
            face = frame[y:y+h,x:x+w]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face,(100,100))
            face = face.reshape(1,100,100,1)
            face = face/255
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            st.write(frame,
                    get_pred_label(np.argmax(model.predict(face), axis=-1)[0]),
                        (200,500),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            st.write(get_pred_label(np.argmax(model.predict(face), axis=-1)[0]))
        
        cv2.imshow("cap",frame)
    if cv2.waitKey(30) == ord("q"):
        break
    
cv2.destroyAllWindows()