import torch


#打开pth文件
# pthfile = r'/home/yscheng/px/multi-label-ood-master-jointenergy/m_saved_models/densenet121-a639ec97.pth'   #.pth文件的路径
# model = torch.load(pthfile, torch.device('cpu'))    #设置在cpu环境下查询
# print('——————————————type——————————————')
# print(type(model))   #查看模型字典长度
# #print('——————————————length——————————————')
# #print(len(model))
# print('——————————————key——————————————')
# for k in model.keys():  #查看模型字典里面的key
#     print(k)
# #print('——————————————value——————————————')
# #for k in model:         #查看模型字典里面的value
# #    print(k,model[k])


"""

#打开npy文件
import numpy as np
file = np.load('/home/yscheng/px/multi-label-ood-master/logits/pascal/imagenet/densenet/in_val.npy')
print(file)
#np.savetxt('.../a/timestamps.txt',file)
"""

# #打开pkl文件
import pickle

path='/home/yscheng/px/datasetttt/coco/multi-label-val2014.pkl'   
	   
f=open(path,'rb')
data=pickle.load(f)

image_path = '/home/yscheng/px/datasetttt/coco/val2014/COCO_val2014_000000001083.jpg'
labels = data.get(image_path)

if labels is not None:
    print("标签：", labels)
else:
    print("未找到该图像的标签。")

# print(data)
# print(len(data))
