# USAGE
# python gather_examples.py --input videos/real.mp4 --output dataset/real --detector face_detector --skip 1
# python gather_examples.py --input videos/fake.mp4 --output dataset/fake --detector face_detector --skip 4

import numpy as np
import argparse
import cv2
import os

# 参数设置
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.95,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=4,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# 人脸检测器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

while True:
	(grabbed, frame) = vs.read()
	
	if not grabbed:
		break
	
	read += 1
	
	# 跳过skip帧
	if read % args["skip"] != 0:
		continue
	
	# opencv dnn 模块预处理，提取blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	
	# 前向传播，得到预测结果
	net.setInput(blob)
	detections = net.forward()
	
	# 检测到脸时
	if len(detections) > 0:
		# 假设一帧只有一个人脸，寻求最高置信度的脸
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		
		# 过滤掉低置信度人脸
		if confidence > args["confidence"]:
			# 得到人脸roi
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			CenterX = (startX + endX)/2
			CenterY = (startY + endY)/2
			face_width = endX - startX
			face_height = endY - startY
			face_width = face_width * 1.5
			face_height = face_height * 1.5
			startX = int(CenterX - (face_width /2))
			endX = int(CenterX + (face_width/2))
			startY = int(CenterY - (face_height /2))
			endY = int(CenterY + (face_height / 2))
			
			# 确保roi 小于帧的尺寸范围
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)
			
			face = frame[startY:endY, startX:endX]
			
			# 保存输出
			p = os.path.sep.join([args["output"],
				"lab_test1_{}.jpg".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

# 清堆栈
vs.release()
cv2.destroyAllWindows()
