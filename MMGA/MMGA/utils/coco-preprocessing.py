# coding: utf-8

from pycocotools.coco import COCO
import argparse
import numpy as np
#import skimage.io as io
import pylab            #PyLab 是一个面向 Matplotlib 的绘图库接口
import os, os.path      #os.path 模块主要用于获取文件的属性
import pickle           #pickle能够实现任意对象与文本之间的相互转化，也可以实现任意对象与二进制之间的相互转化。也就是说，pickle 可以实现 Python 对象的存储及恢复。实现基本的数据序列化和反序列化。
from tqdm import tqdm   #tqdm是一个方便且易于扩展的Python进度条，可以在python执行长循环时在命令行界面实时地显示一个进度提示信息，包括执行进度、处理速度等信息，且可在一定程度上进行定制。

#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

parser = argparse.ArgumentParser(description="Preprocess COCO Labels.")  #创建解析器，description="程序的主要功能是..."

#添加参数
#dataDir='/share/data/vision-greg/coco'
#which dataset to extract options are [all, train, val, test]
#dataset = "all"
parser.add_argument("--dir", type=str, default="/home/yscheng/px/datasetttt/coco/",  #在参数前加上前缀--，即意味着这个参数是可选参数
                    help="where is the coco dataset located.")            #--dir进入某个路径，default是路径默认值，help说明路径含义
parser.add_argument("--save_dir", type=str, default="/home/yscheng/px/multi-label-ood-master/saved_models/coco/",
                    help="where to save the coco labels.")
parser.add_argument("--dataset", type=str, default="all",
                    choices=["all", "train", "val", "test"],
                    help="which coco partition to create the multilabel set" 
                    "for the options [all, train, val, test] default is all")
args = parser.parse_args()  #解析参数


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:  #生成.pkl文件（二进制文件）。常用于保存神经网络训练的模型或者各种需要存储的数据。wb覆盖写
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)  #将obj对象以pkl文件保存起来。dump()将对象转换成字节流，即串行化。HIGHEST_PROTOCOL：表示最高协议
        #f表示保存到的类文件对象,file必须有write()接口，file可以是一个以’w’打开的文件或者是一个StringIO对象，也可以是任何可以实现write()接口的对象。

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:  #rb以二进制形式读取文件中的数据
        return pickle.load(f)   #有了pickle这个对象, 就能对 f(文件) 以读取的形式打开

def wrrite(fname, d):
    fout = open(fname, 'w')
    for i in range(len(d)):
        fout.write(d[i] +'\n')
    fout.close()

def load(fname):
    data = []
    labels = []
    for line in open(fname).readlines():  #是把一个文档的每一行（包含行前的空格，行末加一个\n），作为列表的一个元素，存储在一个list中。每一个行作为list的一个元素。
        l = line.strip().split(' ')#strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                                   #split（‘ ’）: 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串。
        data.append(l[0])
        labels.append(int(l[1]))
    return data,np.array(labels,dtype=np.int32)  #要转换为的数据类型对象


def load_labels(img_names, root_dir, dataset, coco, idmapper):
    labels = {}
    for i in tqdm(range(len(img_names))):   #循环images所有图片
        #print(i, dataset)
        #print(img_names[i], img_names[i][18:-4])
        # Hack to extract the image id from the image name
        if dataset == "val":
            imgIds=int(img_names[i][18:-4])
        else:
            imgIds=int(img_names[i][19:-4])  
        annIds = coco.getAnnIds(imgIds=imgIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        c = []
        for annot in anns:
            c.append(idmapper[annot['category_id']])
        if not c:
            c = np.array(-1)
        labels[root_dir + '/' + img_names[i]] = np.unique(c)

    return labels


def load_image_names(root_dir):
    DIR = root_dir
    #print(DIR)
    img_names = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    return img_names


def load_annotations(dataDir, dataType):
    annFile='%sannotations/instances_%s.json'%(dataDir, dataType)
    
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    
    
    nms=[cat['id'] for cat in cats]
    idmapper = {}
    for i in range(len(nms)):
        idmapper[nms[i]] = i

    return coco, idmapper


root_dir = args.dir + "train2014"
train_img_names = load_image_names(root_dir)
root_dir = args.dir + "val2014"
val_img_names = load_image_names(root_dir)

if args.dataset == "test" or args.dataset == "all":
    root_dir = args.dir + "test2014"
    test_img_names = load_image_names(root_dir)

    d = {}
    for i in range(len(test_img_names)):
        d[i] = root_dir + '/' + test_img_names[i]

    LIST = args.save_dir + 'test2014imgs.txt'
    wrrite(LIST,d)


if args.dataset == "all":
    root_dir = args.dir + "train2014"

    coco, idmapper = load_annotations(args.dir, "train2014")
    labels = load_labels(train_img_names, root_dir, "train", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-train2014") 
    LIST = args.save_dir + "train2014imgs.txt"
    wrrite(LIST, train_img_names)

    root_dir = args.dir + "val2014"

    coco, idmapper = load_annotations(args.dir, "val2014")
    labels = load_labels(val_img_names, root_dir, "val", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-val2014")
    LIST = args.save_dir + "/val2014imgs.txt"
    wrrite(LIST, val_img_names)

elif args.dataset == 'val':

    root_dir = args.dir + "val2014"

    coco, idmapper = load_annotations(root_dir)

    labels = load_labels(val_img_names, root_dir, "val", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-val2014")
    LIST = args.save_dir + "/val2014imgs.txt"
    wrrite(LIST, val_img_names)


elif args.dataset == 'train':
    root_dir = args.dir + "/train2014"

    coco, idmapper = load_annotations(root_dir)

    labels = load_labels(train_img_names, root_dir, "train", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-train2014")
    LIST = args.save_dir + "/train2014imgs.txt"
    wrrite(LIST, train_img_names)



# For image segmentaion 
# converting polygon and RLE to binary mask

#labels = {}
#for i in range(len(imgsname)):
#    print(i)
#    if val == True:
#        imgIds=int(imgsname[i][19:25])
#    else:
#        imgIds=int(imgsname[i][21:27])  
#    annIds = coco.getAnnIds(imgIds=imgIds, iscrowd=None)
#    anns = coco.loadAnns(annIds)
#    for annot in anns:
#        cmask_partial = coco.annToMask(annot)
#
