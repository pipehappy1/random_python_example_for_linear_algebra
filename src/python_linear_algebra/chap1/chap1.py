from PIL import Image  #引入Image模块
from PIL import ImageEnhance  #引入ImageEnhance模块
#  可以合并为from PIL import Image, ImageEnhance

img = Image.open("lena256.jpg")  #读入图像文件lena256.jpg
img.show()  #显示图像

box=(100,100,200,200)
region_img=img.crop(box)
region_img.save("img_region.jpg")
region_img.show()

new_img = img.resize((128,128),Image.BILINEAR)  #改变图像的大小
new_img.save("img_new.jpg")  #保存结果图像
new_img.show()  

rot_img = new_img.rotate(45)  #将图像绕其中心点逆时针旋转45度
rot_img.save("img_rot.jpg")
rot_img.show()  

##rot_img.save("img_rot.bmp")  # 转换图像格式

rot_img.histogram()  #输出图像直方图数据统计结果

#  图像亮度增强
brightness = ImageEnhance.Brightness(img)
bright_img = brightness.enhance(2.0)
bright_img.save("img_bright.jpg")
bright_img.show()

#  图像尖锐化
sharpness = ImageEnhance.Sharpness(img)
sharp_img = sharpness.enhance(7.0)
sharp_img.save("img_sharp.jpg")
sharp_img.show()

#  图像对比度增强
contrast = ImageEnhance.Contrast(img)
contrast_img = contrast.enhance(2.0)
contrast_img.save("img_contrast.jpg")
