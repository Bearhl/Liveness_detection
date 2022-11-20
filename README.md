deploy// 活体检测神经网络结构的定义
dataset// 空，暂时没用
dataset_all// 自己拍的视频，然后用gather_examples.py 生成的人脸图片数据
dataset_new// 在上述文件的基础上，扩充了南航大的视频攻击人脸数据集，最新的模型就是用这个数据集训练的
videos// 这个很关键，最好多多采集实际应用场景里的真、假视频
le.pickle 为二进制文件，保存标签数据，可用可不用 
h5_to_pb.py 从keras的.model或者.h5模型文件转成tensorflow下的.pb模型 亲测可用


gather_examples.py 
作用：
	从拍摄的视频中收集真/假数据集，并保存成人脸图片形式
用法：
	# python gather_examples.py --input videos/csb001_real.mp4 --output dataset_new/real --detector face_detector --skip 1
	# python gather_examples.py --input videos/csb002_fake.mp4 --output dataset_new/fake --detector face_detector --skip 4
	skip代表每隔skip个帧就检测并保存一次人脸，防止数据大量重复


train_liveness.py
作用：
	训练活体检测器，生成活体检测模型
用法：
	# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle
	le.pickle为二进制文件，保存标签数据，可用可不用
	
liveness_demo.py
作用：
	demo程序
用法：
	# python liveness_demo.py --model liveness.model  --detector face_detector