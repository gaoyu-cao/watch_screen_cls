'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:12:06
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:12:07
'''
'''
Descripttion:根据标注的头屏坐标，在原始图像上绘制，用于检测测试标注的对错
version: 
Author: 
Date: 2021-07-13 06:53:21
LastEditors: Please set LastEditors
LastEditTime: 2021-07-15 02:45:10
'''
import cv2

path = '/mnt/sda2/Public_Data/Data_lookscreen/test_images/2021_06_06_09_00_00_D64107836_440114001-5950-7520.jpg'
# 0,868,184,1070,585,301,376,407,532'2,234,132,464  510,444,643,614
img = cv2.imread(path)
img = cv2.resize(img, (720, 1280))

cv2.rectangle(img, (2, 234), (132, 464), (155, 155, 0), thickness=2)  # filled

cv2.rectangle(img, (510, 444), (643, 614), (0, 255, 255), thickness=2)
cv2.imwrite('error_img_2.jpg', img)