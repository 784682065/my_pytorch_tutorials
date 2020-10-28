from PIL import Image
import matplotlib.pyplot as plt
Image.open('PennFudanPed/PNGImages/FudanPed00001.png')

mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
#
# mask.putpalette([
#     0, 0, 0,  # black background
#     255, 0, 0,  # index 1 is red
#     255, 255, 0,  # index 2 is yellow
#     255, 153, 0,  # index 3 is orange
# ])

plt.imshow(mask)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('image')  # 图像题目
plt.show()
