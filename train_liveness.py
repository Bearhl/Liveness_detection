# USAGE
# python train_liveness.py --dataset dataset --model liveness_pro.model --le le.pickle

# 初始化 matplotlib ，保存训练参数图
import matplotlib
matplotlib.use("Agg")

from deploy.livenessnet import LivenessNet
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# 变量输入
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# 初始化learning rate, batch size, epoch
INIT_LR = 1e-3
BS = 32
EPOCHS = 35

# 提取数据集中所有图片 并保存路径到列表中
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# 从文件名中提取标签(fake or real) 再resize所有图片到(32,32)
# 标签加到labels
# 图片数据加到data
for imagePath in imagePaths:

	label = imagePath.split(os.path.sep)[-2]
	# image = cv2.imread(imagePath)
	# print(imagePath, " ", image.shape)
	# if len(image) < 1:
	# 	print("empty!!!hahaha")
	# 	continue
	# cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	# image = cv2.resize(image, (96,96))

	# data.append(image)
	labels.append(label)

# 所有图像数据转成0-1之间的ndarray形式
# data = np.array(data, dtype="float16") / 255.0

# 将fake / real 转成整数形式 进行one-hot编码
# 如[1,0]是fake [0,1]是real 数据类型为float
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
print(len(labels))
print("labels " ,labels)

# 75% 数据训练 25% 数据验证
(trainX, testX, trainY, testY) = train_test_split(imagePaths, labels,
	test_size=0.2, random_state=42)

test_path2ndarry = [cv2.imread(test_path) for test_path in testX]
ndarry2resize = [cv2.resize(img, (32, 32)) for img in test_path2ndarry]
testX = np.array(ndarry2resize, dtype="float16") / 255.0

# keras自带的数据增强 扩大数据集
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 开始训练
print("trainX ", len(trainX))
print("trainY ", len(trainY))
print("[INFO] training network for {} epochs...".format(EPOCHS))
# model = load_model('liveness_pro8.model')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS, transform=True),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)

# 评估神经网络
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# 保存
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# 保存二进制标签数据 后续的 demo 中可以用也可以不用
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# 画图，效果图 ，含有train_loss，val_loss，train_acc，val_acc
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
