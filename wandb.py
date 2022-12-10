
import os
"""
f = open("/home/ubuntu/MSSNET_train_final/MSSNet/datalist/datalist_gopro_test.txt","w")
#file_name = os.listdir("/home/ubuntu/MSSNET_train_final/MSSNet/data_og/blur")
file_name_blur = sorted(os.listdir("/home/ubuntu/MSSNET_train_final/MSSNet/data_og/blur"))
file_name_sharp = sorted(os.listdir("/home/ubuntu/MSSNET_train_final/MSSNet/data_og/sharp"))
"""
"""
#print(file_name[0])
for i in file_name:
    
    if i.split("_")[-1]=="l.png":
        #000111_10_r.png
        sharp_path = os.path.join("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/sharp_split/val/left",i.split(".")[0][:-2]+".png")
        blur_path = os.path.join("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/sum",i)
    elif i.split("_")[-1]=="r.png":
        sharp_path = os.path.join("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/sharp_split/val/right",i.split(".")[0][:-2]+".png")
        blur_path = os.path.join("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/sum",i)
    f.write(sharp_path+" "+blur_path+"\n")
f.close()
"""
"""
#print(file_name[0])
blur_path = "/home/ubuntu/MSSNET_train_final/MSSNet/data_og/blur/"
sharp_path = "/home/ubuntu/MSSNET_train_final/MSSNet/data_og/sharp/"

for i in range(len(file_name_sharp)):
    
    f.write(sharp_path+file_name_sharp[i]+" "+blur_path+file_name_blur[i]+"\n")
f.close()
"""

"""
import os
import shutil
file_path = os.listdir("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/left")

for i in file_path:
    left_path = os.path.join("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/left",i)
    right_path = os.path.join("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/right",i)
    shutil.copy(left_path,"/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/sum/"+i.split(".")[0]+"_l.png")
    shutil.copy(right_path,"/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/val/sum/"+i.split(".")[0]+"_r.png")
"""
"""
#padding
import os
import numpy as np
import cv2
for i in os.listdir("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/test/sum"):

    img = cv2.imread("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/test/sum/"+i)
    
    padded_img = np.pad(img, ((720-img.shape[0], 0),(1280-img.shape[1], 0),(0, 0)), 'constant', constant_values=(4, 6))
    cv2.imwrite("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/test/sum_padding/"+i,padded_img)
"""


"""
#replace
import os
def replace_in_file(file_path, old_str, new_str):
    # 파일 읽어들이기
    fr = open(file_path, 'r')
    lines = fr.readlines()
    fr.close()
    
    # old_str -> new_str 치환
    fw = open(file_path, 'w')
    for line in lines:
        fw.write(line.replace(old_str, new_str))
    fw.close()

# 호출: file1.txt 파일에서 comma(,) 없애기
replace_in_file("/home/ubuntu/MSSNET_train_final/MSSNet/datalist/datalist_kitti_test_crop.txt", "sum_padding", "sum_crop")
"""

#cropping (368*1216*3)
import os
import numpy as np
import cv2
for i in os.listdir("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/test/sum"):

    img = cv2.imread("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/test/sum/"+i)
    
    cropped_img = img[:368,:1216,:]
    #print(cropped_img.shape)
    cv2.imwrite("/home/ubuntu/MSSNET_train_final/MSSNet/dataset/blur_split/test/sum_crop/"+i,cropped_img)
