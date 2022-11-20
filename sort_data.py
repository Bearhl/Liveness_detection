import os
import numpy
import random, shutil

def MoveFilesToNewDir(fileDir, tarDir,rate):
    #sonDirPath = []
    AllDir = os.listdir(fileDir)  # 列出指定路径下的全部文件夹，以列表的方式保存
    # print(AllDir)
    for dir in AllDir:  # 遍历指定路径下的全部文件和文件夹
        sonDirName = os.path.join(fileDir, dir)  # 子文件夹的路径名称
        # print(sonDirName)
        if os.path.isdir(sonDirName):
            #sonDirPath.append(sonDirName)
            pathDir = os.listdir(sonDirName)  # 取图片的原始路径
            filenumber = len(pathDir)
            # print(filenumber)
            picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
            sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
            # print(sample)
            for name in sample:
                oldDir = sonDirName +'\\'+ name
                newDir = tarDir + '\\' + dir
                isExists = os.path.exists(newDir)
                if not isExists:
                    os.makedirs(newDir)
                newTarDir = tarDir +'\\'+dir +'\\'+ name
                print(newTarDir)
                # with os.open()
                print(oldDir, newTarDir)
                shutil.move(oldDir,  newDir)

fileDir = r"E:\PyCharm\Faceliveness\dataset0"      #源图片文件夹路径
valDir = r'E:\PyCharm\Faceliveness\dataset0\train'       #移动到验证集目录路径
MoveFilesToNewDir(fileDir, valDir,0.8)