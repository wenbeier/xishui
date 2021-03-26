import numpy as np
import os
import cv2
import copy

def get_labels():
    with open("coco_label.txt", "r") as f:
        label =[]
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            label.append(line)
            #print(label)
    return label

def find_index(str):
    labels = get_labels()
    for i in range(0,len(labels)):
        if labels[i] == str:
            index = i
    if str not in labels:
        index = -1
    return index

def get_depth(x, y, rgb_depth):
    # rouph function
    x = round(x)
    y = round(y)
    image = cv2.imread(rgb_depth)  # 读取图像
    depth = image[x, y]
    return depth[0]

def yolo_topo_array(img_out, rgb_depth):

    topo_array = np.zeros((len(img_out), 5), dtype=float)
    #基本数组
    for i in range(0, len(img_out)):
        # 修改第一列元素为索引，错误返回-1
        #索引 置信度 x, y,z
        topo_array[i][0] = find_index(str(img_out[i][0]))
        topo_array[i][1] = img_out[i][1]
        topo_array[i][2] = (img_out[i][2][0] + img_out[i][2][2]) / 2
        topo_array[i][3] = (img_out[i][2][1] + img_out[i][2][3]) / 2
        topo_array[i][4] = get_depth(topo_array[i][2], topo_array[i][3], rgb_depth)

    #重复标签规则 x坐标小则编号小
    for i in range(0, len(topo_array)):
        for k in range(i+1, len(topo_array)):
            if topo_array[i][0] == topo_array[k][0]:
                #print('true')
                if topo_array[i][2] < topo_array[k][2]:
                    topo_array[k][0] = topo_array[i][0]+80
                else:
                    topo_array[i][0] = topo_array[k][0] + 80
    return topo_array



'''
img_out_1=[('chair', '74.34', (111.53564453125, 478.2809143066406, 197.55352783203125, 260.8075866699219)),
          ('mouse', '75.51', (255.5177764892578, 374.8509216308594, 15.9299955368042, 14.70186710357666)),
          ('keyboard', '79.5', (406.0572204589844, 386.55535888671875, 93.32878875732422, 26.0970516204834)),
          ('bottle', '81.07', (465.2334899902344, 354.26446533203125, 15.568439483642578, 57.528133392333984)),
          ('tvmonitor', '95.93', (212.64617919921875, 285.1939392089844, 79.47523498535156, 86.1717758178711)),
          ('tvmonitor', '97.25', (402.732666015625, 301.6360168457031, 123.64592742919922, 124.32866668701172)),
          ('chair', '97.81', (421.1565856933594, 499.93475341796875, 110.76861572265625, 199.34896850585938)),
          ('person', '98.53', (529.2282104492188, 299.9497375488281, 152.8038330078125, 584.7741088867188)),
          ('person', '98.54', (137.48951721191406, 378.5865173339844, 174.62451171875, 445.9444885253906)),
          ('keyboard', '99.59', (199.2576904296875, 357.65887451171875, 88.77013397216797, 29.83479118347168))]

xx=yolo_topo_array(img_out_1, '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/'
                              '1341846313.592088.png')
print(xx)
'''


