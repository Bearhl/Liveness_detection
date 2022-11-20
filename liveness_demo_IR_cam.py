# USAGE
# python liveness_demo.py --model liveness.model  --detector face_detector

#from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import os

# 参数设置
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, default="face_detector",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 人脸检测器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 加载模型

labels = ["fake", "real"]

print("[INFO] starting video stream...")
#vs = VideoStream(src="test.mp4").start()
#vs = VideoStream(“test.mp4”).start()
v_rgb = cv2.VideoCapture(0)
#v_ir = cv2.VideoCapture(2)

# size 被按比例压缩到最大width为600
# r = float(int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)) / int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)))
# size = (600, int(600 * r))
size = (int(v_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)))

writer = cv2.VideoWriter("out3.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15.0, size)
# print("(width, height)", size)

# count = fake_num = real_num = 0
# judge = ""

while True:

	ret_rgb, frame_rgb = v_rgb.read()
	# ret_ir, frame_ir = v_ir.read()
	# if not ret_ir:
		# continue
	# else:
	# 	count +=1
	#
	# # 10帧为窗口判断真假
	# if count % 11 ==0:
	# 	count = 0
	# elif count is 10:
	# 	if fake_num >= 9:
	# 		judge = "fake"
	# 	if real_num >= 9:
	# 		judge = "real"
	# 	real_num = 0
	# 	fake_num = 0
	# print("count ",count, "fake_num ", fake_num, "real_num ", real_num, "judge ", judge)
	# size 被按比例压缩到最大width为600
	# frame = cv2.resize(frame,size)

	# opencv dnn 模块预处理，提取blob
	# (h, w) = frame_ir.shape[:2]
	# blob = cv2.dnn.blobFromImage(cv2.resize(frame_ir, (300, 300)), 1.0,
		# (300, 300), (104.0, 177.0, 123.0))

	# 前向传播，得到预测结果
	# net.setInput(blob)
	# detections = net.forward()

	# 检测到人脸时
	for i in range(0, detections.shape[2]):
		# 置信度
		confidence = detections[0, 0, i, 2]
  
		# 过滤掉低置信度人脸
		if confidence > args["confidence"]:
			# 得到人脸roi
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}".format("real")
			cv2.putText(frame_ir, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame_ir, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
    
	cv2.imshow("Frame", frame_ir)
	writer.write(frame_rgb)
 
	# 按q停止
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# 清堆栈
cv2.destroyAllWindows()
v_rgb.release()
# v_ir.release()
writer.release()