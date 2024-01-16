import cv2
import mediapipe as mp
import time
 
cap = cv2.VideoCapture(0)  # 调用镜头
wcap = cap.set(3, 800)     # 设置相框大小
hcap = cap.set(4, 800)
 
mpHands = mp.solutions.hands  # 使用mediapipe 手部模型
hands = mpHands.Hands()
drawmp = mp.solutions.drawing_utils  # 画线
topIds = [4, 8, 12, 16, 20]     #5根手指的指尖
pTime = 0
while True:
    success, img = cap.read()  # cap.read()会返回两个值：Ture或False 和 帧
    if success:
        list = []
        #opencv调用相机拍摄的图像格式是BGR,得转化为RGB格式便于图像处理
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks) #打印手部21个点的坐标信息
        if result.multi_hand_landmarks:
            for handlm in result.multi_hand_landmarks:
                # print(handlm) #打印坐标信息
                drawmp.draw_landmarks(img, handlm, mpHands.HAND_CONNECTIONS) #将21个点连线
                for id, lm in enumerate(handlm.landmark):
                    h, w, c = img.shape  #图像的长、宽、通道
                    cx, cy = int(lm.x * w), int(lm.y * h)  #坐标cx,cy转为整数
                    list.append([id, cx, cy])    #记录每点坐标
            if len(list) != 0:
                # 判断左右手
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0] # 仅取第一个检测到的手
                    if hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x < hand_landmarks.landmark[
                        mpHands.HandLandmark.THUMB_TIP].x:
                        print("右手")
                    else:
                        print("左手")
    cv2.imshow("images", img)
    if cv2.waitKey(1) & 0xff == 27:  #按‘ESC’键退出
        break