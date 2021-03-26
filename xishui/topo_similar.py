import numpy as np
import gist_cossim
from darknet_images import main
from yolo_topo import yolo_topo_array

def vector_similar(vec1,vec2,t):
    vec1_n = np.linalg.norm(vec1, ord=2)
    vec2_n = np.linalg.norm(vec2, ord=2)  # vector norm
    t_n = np.linalg.norm(t, ord=2)

    if vec2_n * vec1_n == 0:
        vec2_n = 0.0001
        vec1_n = 0.0001
    alpha = vec1_n / vec2_n
    theta = np.arccos(np.dot(vec1, vec2) / (vec1_n * vec2_n))
    omigax = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    omigay = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    omigaz = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    sigma = np.cos(theta)
    sigma_n = np.sin(theta)
    R = np.array([[sigma + (1 - sigma) * omigax * omigax, (1 - sigma) * omigax * omigay - omigaz * sigma_n,
                     (1 - sigma) * omigax * omigaz + omigay * sigma_n],
                     [(1 - sigma) * omigax * omigay + omigaz * sigma_n, sigma + (1 - sigma) * omigay * omigay,
                      (1 - sigma) * omigaz * omigay - omigax * sigma_n],
                     [(1 - sigma) * omigax * omigaz - omigay * sigma_n, (1 - sigma) * omigax * omigaz + omigay * sigma_n,
                      sigma + (1 - sigma) * omigaz * omigaz]])
    R_n = np.linalg.norm(R, ord=2)
    S = np.exp(-0.5 * np.abs(1 - alpha * R_n - t_n))
    print(S)
    return S


def match_point(topo_1, topo_2):
    match_point_list = np.zeros((len(topo_1), 2), dtype=int)
    for i in range(0, len(topo_1)):
        #print("cccc")
        for k in range(0, len(topo_2)):
            #print("ccc")
            if topo_1[i][0] == topo_2[k][0]:
                #print("cc")
                match_point_list[i][0] = i
                match_point_list[i][1] = k
    return match_point_list


def topo_points_pair(topo_1, topo_2, match_list):
    points_pair = np.zeros((len(match_list),2,3), dtype=float)
    for i in range(0, len(match_list)):
        points_pair[i][0][0] = topo_1[match_list[i][0]][2]
        points_pair[i][0][1] = topo_1[match_list[i][0]][3]
        points_pair[i][0][2] = topo_1[match_list[i][0]][4]

        points_pair[i][1][0] = topo_2[match_list[i][1]][2]
        points_pair[i][1][1] = topo_2[match_list[i][1]][3]
        points_pair[i][1][2] = topo_2[match_list[i][1]][4]
    return points_pair


def sim_topo(points_pair):
    Sim_sum = 0
    num = 0
    for i in range(0, len(points_pair)):
        for k in range(i+1, len(points_pair)):
            a1 = np.array([points_pair[i][0][0], points_pair[i][0][1], points_pair[i][0][2]])
            a2 = np.array([points_pair[i][1][0], points_pair[i][1][1], points_pair[i][1][2]])

            b1 = np.array([points_pair[k][0][0], points_pair[k][0][1], points_pair[k][0][2]])
            b2 = np.array([points_pair[k][1][0], points_pair[k][1][1], points_pair[k][1][2]])
            t = a1 - a2
            vec1 = a1 - b1
            vec2 = a2 - b2
            #print(vec1, vec2, t)
            Sim_sum += vector_similar(vec1, vec2, t)
            #print(Sim_sum)
            num += 1
    return Sim_sum


if __name__ == '__main__':
    '''
    img1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png'
    depth_img_1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/1341846313.592088.png'
    yolo1 = main(img1)
    topo_1 = yolo_topo_array(yolo1, depth_img_1)
    
    img2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.592026.png'
    depth_img_2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/1341846313.654212.png'
    yolo2 = main(img2)
    topo_2 = yolo_topo_array(yolo2, depth_img_2)
    

    img1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png'
    depth_img_1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/1341846313.592088.png'
    yolo1 = main(img1)
    topo_1 = yolo_topo_array(yolo1, depth_img_1)

    img2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846319.786397.png'
    depth_img_2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/1341846319.786406.png'
    yolo2 = main(img2)
    topo_2 = yolo_topo_array(yolo2, depth_img_2)
    '''
    img1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png'
    depth_img_1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/1341846313.592088.png'
    yolo1 = main(img1)
    topo_1 = yolo_topo_array(yolo1, depth_img_1)

    img2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846314.057816.png'
    depth_img_2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/1341846314.057840.png'
    yolo2 = main(img2)
    topo_2 = yolo_topo_array(yolo2, depth_img_2)
    
    images = ['1341846313.592026',
'1341846313.654184',
'1341846313.686156',
'1341846313.721918',
'1341846313.753994',
'1341846313.789969',
'1341846313.822075',
'1341846313.853928',
'1341846313.890011',
'1341846313.922055',
'1341846313.957927',
'1341846313.990058',
'1341846314.022042',
'1341846314.057816',
'1341846314.089801',
'1341846314.122037',
'1341846314.157989' ,
'1341846314.190094' ,
'1341846314.225969' ,
'1341846314.257923' ,
'1341846314.290052' ,
'1341846314.325981',
'1341846314.357905' ,
'1341846314.389899' ,
'1341846314.427905' 
'1341846314.493987',
'1341846314.526085' ,
'1341846314.558292' ]

    depth = ['1341846313.592088.png',
'1341846313.654212.png',
'1341846313.686172.png',
'1341846313.721932.png',
'1341846313.754052.png',
'1341846313.789985.png',
'1341846313.822093.png',
'1341846313.853940.png',
'1341846313.890055.png',
'1341846313.922606.png',
'1341846313.957943.png',
'1341846313.990646.png',
'1341846314.022058.png',
'1341846314.057840.png',
'1341846314.089812.png',
'1341846314.122046.png',
'1341846314.158005.png',
'1341846314.190124.png',
'1341846314.226054.png',
'1341846314.257955.png',
'1341846314.290094.png',
'1341846314.326013.png',
'1341846314.357937.png',
'1341846314.389929.png',
'1341846314.427105.png',
'1341846314.493999.png',
'1341846314.526098.png',
'1341846314.558316.png']

    for i in  = 0
    img1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/'+str(images[i])+'.png'
    depth_img_1 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/'+str(depth[i])
    print(img1)
    yolo1 = main(img1)
    topo_1 = yolo_topo_array(yolo1, depth_img_1)

    img2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb/' + str(images[i+1])+'.png'
    depth_img_2 = '/home/wan/wyyslam/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/depth/'+str(depth[i+1])
    yolo2 = main(img2)
    topo_2 = yolo_topo_array(yolo2, depth_img_2)
    '''
        #print(topo_1)
        #print(topo_2)
    match_list = match_point(topo_1, topo_2)
        #print(match_list)
    pair = topo_points_pair(topo_1, topo_2, match_list)
        #print(pair)
    SSS = sim_topo(pair)

    with open('test.txt', 'w') as f:  # 打开test.txt   如果文件不存在，创建该文件。
        f.write(str(SSS))  # 把变量var写入test.txt。这里var必须是str格式，如果不是，则可以转一下。

        #print("the similar of two images is : ")
        #print(SSS)