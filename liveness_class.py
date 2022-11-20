from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os

# ---------------加载活体检测地址---------------
# 1.活体检测模型路径
liveness_model_path = 'E:\\PyCharm\\Faceliveness\\dataset_simple\\liveness_edge15.model'
# 2.人脸检测检测器路径
face_detection_path = 'E:\\PyCharm\\Faceliveness\\face_detector'
# 3.默认的confidence阈值
confidence_dafult = 0.6
# ---------------活体检测地址 End---------------

# ---------------加载活体检测模型---------------
protoPath = os.path.sep.join([face_detection_path, "deploy.prototxt"])
modelPath = os.path.sep.join([face_detection_path, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
model = load_model(liveness_model_path)
# ---------------活体检测模型 End---------------

# ----------初始化活体检测判断窗口参数----------
timeline = 0    # 时间戳计数
flag = 0
count = 0
label_list = []
labels = ["Fake", "Real"]
# -----------活体检测判断窗口参数 End-------------

class liveness:
    def __init__(self, input_img):
        self.cap = cv2.VideoCapture(0)
        self.sign = 0
        self.frame =input_img

    # 类输入图片，输出人脸的位置的四维坐标以及判断真假结果
    def liveness_detection(self):
        global timeline, flag, count, label_new, table
        # size 被按比例压缩到最大width为600
        # 保存视频尺寸调整及初始化
        r = float(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        size = (600, int(600 * r))
        writer = cv2.VideoWriter("record.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15.0, size)
        print("(width, height)", size)

        # 将传入图片传入模型，判断活体
        # 检测是否有人脸
        frame = cv2.resize(self.frame, size)
        # opencv dnn 模块预处理，提取blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        # 前向传播，得到预测结果
        net.setInput(blob)
        detections = net.forward()

        # 检测到人脸时
        for i in range(0, detections.shape[2]):
            # 置信度
            confidence = detections[0, 0, i, 2]
            # 过滤掉低置信度人脸
            if confidence > confidence_dafult:
                    # 得到人脸roi
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                table = [startX, startY, endX, endY]
                CenterX = (startX + endX) / 2
                CenterY = (startY + endY) / 2
                face_width = endX - startX
                face_height = endY - startY
                face_width = face_width * 1.5
                face_height = face_height * 1.5
                big_startX = int(CenterX - (face_width / 2))
                big_endX = int(CenterX + (face_width / 2))
                big_startY = int(CenterY - (face_height / 2))
                big_endY = int(CenterY + (face_height / 2))
                # 确保roi 小于帧的尺寸范围
                if big_startX < 0 or big_startY < 0 or big_endX > w or big_endY > h:
                        continue

                # 得到人脸的坐标并resize为96*96
                face = frame[big_startY:big_endY, big_startX:big_endX]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (96, 96))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # 前向传播liveness_edge15.model 得到预测结果
                preds = model.predict(face)[0]
                j = np.argmax(preds)

                # 以下为对活体的判断及窗口代码
                # 检验到人脸则count计数加一
                if j == 1:
                    count += 1

                # 每8帧是一个计时单位，若8帧里面超过6帧为真则返回真
                if timeline == 8:
                    if count >= 6:
                        label = 'Real'
                    else:
                        label = 'Fake'
                    label_list.append(label)
                    count = 0
                    timeline = 0
                timeline += 1

                # 3个计数单位后进行判断
                # 8*3 = 24
                # 这样可以降低错误率一丢丢，另一种思路就是24帧中有18帧即可
                if len(label_list) == 3:
                    if label_list.count('Real') >= label_list.count('Fake'):
                        label_new = 'Real'
                    else:
                        label_new = 'Fake'
                    del label_list[:]
                    # 清空label_list缓存
                    self.sign = 1
                    # 判断完成的标志

                # 设定阈值，当未检测到人一段时间后，重新初始化参数
                if (confidence <= 0.072) or (confidence >= 0.12):
                    # print('没发现人！！')
                    flag = flag + 1
                    if flag >= 8:
                        self.sign = 0
                        timeline = 0
                        count = 0
                        flag = 0
                else:
                    flag = 0
            writer.write(frame)
            return label_new, table

    # 以下为判断窗口的函数，需要传入人脸位置的四维参数
    # 判断完成后对判断结果进行分析，Real则返回绿色框框，Fake则返回红色框框
    def liveness_windows(self, table):
        global timeline
        startX = table[0]
        startY = table[1]
        endX = table[2]
        endY = table[3]
        if self.sign == 1:
            label = "{}".format(label_new)
            # print(label)
            if label == 'Real':
                cv2.putText(self.frame, label_new, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            else:
                cv2.putText(self.frame, label_new, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        else:
            t = int(timeline / 2) + 1
            loading = "loading{}".format('.' * t)
            label = loading
            cv2.putText(self.frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
