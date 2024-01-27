import cv2
import mediapipe as mp
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv("dataset/data.csv")
X = data.iloc[:, 0:42].values
y = data.iloc[:, 42].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
clf = svm.SVC(kernel='rbf', C=10, gamma=0.01)
clf.fit(X_train_scaled, y)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开相机")
    exit()

cap.set(3, 800)
cap.set(4, 800)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
drawmp = mp.solutions.drawing_utils
topIds = [4, 8, 12, 16, 20]
pTime = 0
while True:
    success, img = cap.read()
    if success:
        finger_list = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handlm in result.multi_hand_landmarks:
                drawmp.draw_landmarks(img, handlm, mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handlm.landmark):
                    h, w, c = img.shape
                    cx, cy = lm.x,lm.y
                    finger_list.extend([cx, cy])
            if len(finger_list) == 21 * 2:  # 检查手部关键点坐标数量是否正确
                X_test_scaled = scaler.transform([np.array(finger_list)])
                y_pred = clf.predict(X_test_scaled)
                cv2.putText(img, str(y_pred[0]), (40, 350), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    cv2.imshow("images", img)
    if cv2.waitKey(1) & 0xff == 27:
        break
