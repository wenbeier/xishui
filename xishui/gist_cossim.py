import os
import cv2
import sys

sys.path.append("./img_gist_feature/")

from utils_gist import *
from util__base import *
from util__cal import *


def get_img_gist_feat(s_img_url):
    gist_helper = GistUtils()
    np_img = cv2.imread(s_img_url, -1)
    np_gist = gist_helper.get_gist_vec(np_img, mode="rgb")
    np_gist_L2Norm = np_l2norm(np_gist)
    return np_gist_L2Norm

def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    np_img_gist_a = get_img_gist_feat(s_img_url_a)
    np_img_gist_b = get_img_gist_feat(s_img_url_b)

    f_img_sim = np.inner(np_img_gist_a, np_img_gist_b)
    return f_img_sim

# 获取两幅图像的gist余弦相似度
def imgs_gist_cosim( img1 , img2):
    O_IN = {}
    O_IN['s_img_url_a'] = str(img1)
    O_IN['s_img_url_b'] = str(img2)
    sim = proc_main(O_IN)
    return sim
    # cos similar

# 遍历图片
def read_directory(directory_name):
    imagelist = []
    for filename in os.listdir(directory_name):
        if filename.lower().endswith(('.png', '.jpg')):
            imagelist.append(os.path.join(directory_name, filename))
    #print(imagelist)
    return imagelist


if __name__ == "__main__":
    # images dir/path
    image_path = "/home/wan/图片/wallweaper"
    image_set = read_directory(image_path)
    #images_similar = []
    for i in range(0, len(image_set)-1):
        sim0 = imgs_gist_cosim(image_set[i], image_set[i+1])
        print(float(sim0))
        #images_similar.append(float(sim0))
        '''
        if float(sim0) > mouzhi
        i 添加到数组里
        
        '''



