'''
Description: build TV Logo dataset, lable num: 0, <object-class> <x_center> <y_center> <width> <height>
Author: Hejun Jiang
Date: 2021-05-11 10:29:38
LastEditTime: 2021-05-18 14:35:20
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import cv2  # h,w,c
import shutil
import random
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--finetune', action='store_true', help='show a done picture for finetune, to ensure the parameters is satisfied')
parser.add_argument('--imagesize', type=list, default=[640, 480], help='the size h,w of builded images')
parser.add_argument('--logo_size_ratio', type=list, default=[0.04, 0.08], help='the ratio of logo height size for background')
parser.add_argument('--logo_rotate', type=float, default=0.2, help='the ratio of rotate logo in background')
parser.add_argument('--img_add_noise', type=float, default=0, help='the ratio of image add noise')
parser.add_argument('--sp_noise_ratio', type=list, default=[0.005, 0.02], help='the parameter of sp noise')
parser.add_argument('--gauss_noise_ratio', type=list, default=[0.04, 0.07], help='the parameter of gauss noise')
parser.add_argument('--datasetdir', type=str, default='./dataset', help='the dir of dataset; background, tvlogos must in here')
parser.add_argument('--trainnum', type=int, default=2048, help='the number of train images for building')
parser.add_argument('--valnum', type=int, default=512, help='the number of val images for building')
parser.add_argument('--testnum', type=int, default=1024, help='the number of test images for building')
parser.add_argument('--projectname', type=str, default='tvlogo', help='the name of project')

conf = parser.parse_args()

imageType = ['jpg', 'png', 'jpeg', 'ico']
rotateType = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]  # logo 进行旋转
datasetdir = os.path.abspath(conf.datasetdir)
backDir = os.path.join(datasetdir, 'background')
logoDir = os.path.join(datasetdir, 'tvlogos')
fakeDir = os.path.join(datasetdir, 'fakes')
yamlPath = os.path.join(datasetdir, 'coco_' + conf.projectname + '.yaml')
if os.path.isfile(yamlPath):
    os.remove(yamlPath)

imagesDir = os.path.join(datasetdir, 'images')
if os.path.isdir(imagesDir):
    shutil.rmtree(imagesDir)
trainImDir = os.path.join(imagesDir, 'train_logo')
valImDir = os.path.join(imagesDir, 'val_logo')
testImDir = os.path.join(imagesDir, 'test_logo')
os.makedirs(trainImDir)
os.makedirs(valImDir)
os.makedirs(testImDir)

labelsDir = os.path.join(datasetdir, 'labels')
if os.path.isdir(labelsDir):
    shutil.rmtree(labelsDir)
trainLaDir = os.path.join(labelsDir, 'train_logo')
valLaDir = os.path.join(labelsDir, 'val_logo')
testLaDir = os.path.join(labelsDir, 'test_logo')
os.makedirs(trainLaDir)
os.makedirs(valLaDir)
os.makedirs(testLaDir)

assert conf.logo_size_ratio[1] > conf.logo_size_ratio[0], 'logo_size_ratio[1] must bigger than logo_size_ratio[0]'
assert conf.sp_noise_ratio[1] > conf.sp_noise_ratio[0], 'sp_noise_ratio[1] must bigger than sp_noise_ratio[0]'
assert conf.gauss_noise_ratio[1] > conf.gauss_noise_ratio[0], 'gauss_noise_ratio[1] must bigger than gauss_noise_ratio[0]'
print('*******************build tv logo dataset for yolov5, please use --finetune first*******************')
print('finetune:', conf.finetune)
print('imagesize:', conf.imagesize)
print('logo_size_ratio:', conf.logo_size_ratio)
print('logo_rotate:', conf.logo_rotate)
print('img_add_noise:', conf.img_add_noise)
print('sp_noise_ratio:', conf.sp_noise_ratio)
print('gauss_noise_ratio:', conf.gauss_noise_ratio)
print('trainnum:', conf.trainnum)
print('valnum:', conf.valnum)
print('testnum:', conf.testnum)
print('projectname:', conf.projectname)
print('imageType:', imageType)
print('rotateType:', rotateType)
print('datasetdir:', datasetdir)
print('backDir:', backDir)
print('logoDir:', logoDir)
print('yamlPath:', yamlPath)
print('trainImDir:', trainImDir)
print('valImDir:', valImDir)
print('testImDir:', testImDir)
print('trainLaDir:', trainLaDir)
print('valLaDir:', valLaDir)
print('testLaDir:', testLaDir)


def getPathList(dir):
    pathlis, objlis = [], set()
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.split('.')[-1] in imageType:
                pathlis.append(os.path.join(root, name))
                objlis.add(os.path.basename(root))

    return pathlis, list(objlis)


def saveYaml(objlis):
    f = open(yamlPath, 'w', encoding='utf-8')
    f.write('# please revise parameters train/val/test if their dir is moved\n')
    f.write('train: '+trainImDir+'\n')
    f.write('val: '+valImDir+'\n')
    f.write('test: '+testImDir+'\n\n')
    f.write('nc: ' + str(len(objlis)) + '\n')
    nstr = ''
    for name in objlis:
        nstr += '\'' + name + '\'' + ','
    if len(nstr) > 0:  # 去除最后一个','
        nstr = nstr[:-1]
    f.write('names: [' + nstr + ']\n')
    f.close()


def gaussValue(scale):
    while True:
        r = random.gauss(scale[0], scale[1])
        if r >= scale[0] and r <= scale[1]:
            return r


def getbox(bimg, rsimg, addedboxs):
    hlen, wlen = rsimg.shape[0], rsimg.shape[1]
    for i in range(1000):
        h = random.randint(0, bimg.shape[0] - hlen)  # 随机左顶点位置
        w = random.randint(0, bimg.shape[1] - wlen)
        ch = h + hlen // 2  # 中心点位置
        cw = w + wlen // 2
        isin = False
        for box in addedboxs:
            if abs(2*(box[0] - ch)) < box[2] + hlen or abs(2*(box[1] - cw)) < box[3] + wlen:
                isin = True
                break
        if not isin:
            return [h, w, ch, cw, hlen, wlen]
    return []  # rand 1000次未出来，代表box差不多占满了


def sp_noise(image):  # 椒盐
    prob1 = random.uniform(conf.sp_noise_ratio[0], conf.sp_noise_ratio[1])  # prob*2
    prob2 = 1 - prob1
    output = np.random.random(image.shape)
    image[output < prob1] = 0
    image[output > prob2] = 255
    return image


def gauss_noise(image):  # 高斯
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, random.uniform(conf.gauss_noise_ratio[0], conf.gauss_noise_ratio[1]), image.shape)
    out = image + noise
    out[out < 0] = 0
    out[out > 1] = 1
    out = np.uint8(out * 255)
    return out


def noiseRand(image):
    addnoise = [sp_noise, gauss_noise]
    rand = random.randint(0, 1)
    img = image
    if random.random() < conf.img_add_noise:
        img = random.choice(addnoise)(image)
    return img


def buildfull(backList, logoList, fakeList, num, imgDir, lableDir, objlis):
    print('imgDir:', imgDir, 'num:', num)
    for i in range(num):  # 一次一张背景图
        if (i + 1) % 100 == 0:
            print('      builded:', i + 1)

        bimg = cv2.imread(random.choice(backList))
        bimg = cv2.resize(bimg, (conf.imagesize[1], conf.imagesize[0]))

        imgs = []
        maxh, maxw = 0, 0
        totalnum = min(len(logoList), len(fakeList))
        slis = random.choices(logoList, k=totalnum)
        flis = random.choices(fakeList, k=totalnum)
        for file in slis+flis:
            if file.endswith('.ico'):  # ico
                simg = cv2.cvtColor(np.asarray(Image.open(file)), cv2.COLOR_RGB2BGR)
                simg[simg == 0] = 255  # 黑转白
            else:
                simg = cv2.imread(file)  # 需保证签名图中白底
            r = gaussValue(conf.logo_size_ratio)  # 缩放
            rsimg = cv2.resize(simg, (int(bimg.shape[0] * r * simg.shape[1] / simg.shape[0]), int(bimg.shape[0] * r)))

            if rsimg.shape[0] > maxh:
                maxh = rsimg.shape[0]
            if rsimg.shape[1] > maxw:
                maxw = rsimg.shape[1]
            dirname = os.path.basename(os.path.dirname(file))
            if file in slis:
                imgs.append([rsimg, objlis.index(dirname)])
            else:
                imgs.append([rsimg])
        hnum, wnum, j = bimg.shape[0]//maxh, bimg.shape[1]//maxw, 0

        random.shuffle(imgs)
        f = open(os.path.join(lableDir, conf.projectname + "_" + str(i) + '.txt'), 'w', encoding='utf-8')
        for hi in range(hnum):  # 最多hnum*wnum个粘贴
            for wi in range(wnum):
                if j < len(imgs):  # 同时不能超过len(imgs)个
                    centerh, centerw = maxh*hi+maxh//2, maxw*wi+maxw//2
                    vertexh, vertexw = centerh - imgs[j][0].shape[0]//2, centerw-imgs[j][0].shape[1]//2
                    box = [vertexh, vertexw, centerh, centerw, imgs[j][0].shape[0], imgs[j][0].shape[1]]

                    rsimggray = cv2.cvtColor(imgs[j][0], cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(rsimggray, 235, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)  # 颠倒黑白，白字

                    mask_inv = cv2.erode(mask_inv, np.ones((2, 2), np.uint8), iterations=1)  # 腐蚀一下白边
                    mask = cv2.bitwise_not(mask_inv)  # 腐蚀之后，mask也要相应改变

                    roi = bimg[box[0]: imgs[j][0].shape[0] + box[0], box[1]: imgs[j][0].shape[1] + box[1]]  # 抠图
                    rsimgbg = cv2.bitwise_and(roi, roi, mask=mask)  # 当mask!=0,roi和roi按位与;当mask==0，为0;黑字
                    rsimgfg = cv2.bitwise_and(imgs[j][0], imgs[j][0], mask=mask_inv)  # 当mask!=0,rsimg和rsimg按位与;当mask==0，为0；彩（黑）字
                    bimg[box[0]: imgs[j][0].shape[0] + box[0], box[1]: imgs[j][0].shape[1] + box[1]] = cv2.add(rsimgbg, rsimgfg)  # 彩（黑）字覆盖黑字，在背景图中

                    if len(imgs[j]) >= 2:
                        f.write(
                            '%d %.6f %.6f %.6f %.6f\n' %
                            (imgs[j][1],
                             box[3] / bimg.shape[1],
                             box[2] / bimg.shape[0],
                             imgs[j][0].shape[1] / bimg.shape[1],
                             imgs[j][0].shape[0] / bimg.shape[0]))
                    j += 1
        f.close()
        bimg = noiseRand(bimg)
        if conf.finetune:
            cv2.imshow('finetune', bimg)
            cv2.waitKey(0)
            exit(0)
        cv2.imwrite(os.path.join(imgDir, conf.projectname + "_" + str(i) + '.jpg'), bimg)


if __name__ == '__main__':
    backList, _ = getPathList(backDir)
    logoList, objlis = getPathList(logoDir)
    fakeList, _ = getPathList(fakeDir)
    buildfull(backList, logoList, fakeList, conf.trainnum, trainImDir, trainLaDir, objlis)
    buildfull(backList, logoList, fakeList, conf.valnum, valImDir, valLaDir, objlis)
    buildfull(backList, logoList, fakeList, conf.testnum, testImDir, testLaDir, objlis)
    saveYaml(objlis)
    print('build dataset done')
