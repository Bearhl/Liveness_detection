# USAGE
# python liveness_demo.py --model liveness.model  --detector face_detector

# from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# 参数设置
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default = 'liveness_large0.model', required=True,
                help="path to trained model")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 人脸检测器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 加载模型
print("[INFO] loading liveness detector...")
model = load_model(args["model"])

labels = ["fake", "real"]

print("[INFO] starting video stream...")
# vs = VideoStream(src="test.mp4").start()
# vs = VideoStream(“test.mp4”).start()
vs = cv2.VideoCapture(0)

# size 被按比例压缩到最大width为600
r = float(int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)) / int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)))
size = (600, int(600 * r))
# size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

writer = cv2.VideoWriter("model6_out3.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15.0, size)
print("(width, height)", size)

# count = fake_num = real_num = 0
# judge = ""

while True:
    
    ret, frame = vs.read()
    if not ret:
        continue
    # else:
    # 	count +=1
    #
    # # 10帧为窗口判断真假
    # if count % 11 ==0:
    # 	count = 0
    # elif count is 10:+
    
    # 	if fake_num >= 9:
    # 		judge = "fake"
    # 	if real_num >= 9:
    # 		judge = "real"
    # 	real_num = 0
    # 	fake_num = 0
    # print("count ",count, "fake_num ", fake_num, "real_num ", real_num, "judge ", judge)
    # size 被按比例压缩到最大width为600
    frame = cv2.resize(frame, size)
    
    # opencv dnn 模块预处理，提取blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # 前向传播，得到预测结果
    net.setInput(blob)
    detections = net.forward()
    
    # 检测到人脸时
    for i in range(0, detections.shape[2]):
        # 置信度
        confidence = detections[0, 0, i, 2]
        
        # 过滤掉低置信度人脸
        if confidence > args["confidence"]:
            # 得到人脸roi
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            CenterX = (startX + endX) / 2
            CenterY = (startY + endY) / 2
            face_width = endX - startX
            face_height = endY - startY
            face_width = face_width * 1.5
            face_height = face_height * 1.5
            big_startX = int(CenterX - (face_width / 2))
            big_endX = int(CenterX + (face_width / 2))
            big_startY = int(CenterY -   (face_height / 2))
            big_endY = int(CenterY + (face_height / 2))
            
            # 确保roi 小于帧的尺寸范围
            if big_startX < 0 or big_startY < 0 or big_endX > w or big_endY > h:
                continue

            if face_width >face_height:
                continue
            # frame[startY:endY, startX:endX] = (0,0,0)
            # resize到32*32，与神经网络的输入尺寸一致
            face = frame[big_startY:big_endY, big_startX:big_endX]
            if face.size == 0:
                continue
            face = cv2.resize(face, (96,96))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            
            # 前向传播liveness.model 得到预测结果
            preds = model.predict(face)[0]
            
            j = np.argmax(preds)
            print(preds, j)
            label = labels[j]
            
            # print("label", label, "j", type(j))
            # j = int(j)
            # if j is 0:
            # 	fake_num += 1
            # elif j is 1:
            # 	real_num +=1
            
            # 输出结果
            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            
    # frame = cv2.resize(frame, (1080, 720))
    cv2.imshow("Frame", frame)
    
    writer.write(frame)
    
    # 按q停止
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
# 清堆栈
cv2.destroyAllWindows()
vs.release()
writer.release()