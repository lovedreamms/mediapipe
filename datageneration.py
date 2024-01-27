import mediapipe as mp
import cv2
import glob
from tqdm import tqdm
#读取image路径下的所有照片
images = glob.glob("dataset/*/*.jpg")
mpHands = mp.solutions.hands 
hands = mpHands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5)
#使用模型生成检测数据并保存为一个文件
list = []
for path in tqdm(images):
    list.append([])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            # print(handlm) #打印坐标信息
            for id, lm in enumerate(handlm.landmark):
                h, w, c = image.shape  #图像的长、宽、通道
                cx, cy = lm.x, lm.y  #坐标cx,cy转为整数
                list[-1].extend([cx, cy])    #记录每点坐标
        list[-1].append(path.split("\\")[-2][4:])
    else:
        print(path)
#把二维的数据保存为csv文件
import csv
with open('dataset/data.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(list)

