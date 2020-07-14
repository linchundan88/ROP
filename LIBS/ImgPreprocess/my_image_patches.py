import cv2
import numpy as np

from LIBS.ImgPreprocess.my_patches_based_seg import extract_patch_non_overlap

'''
# or tf.extract_image_patches(images,ksizes, strides,rates,padding,name=None)

image:输入图像的tesnsor，必须是[batch, in_rows, in_cols, depth]类型
ksize:滑动窗口的大小，长度必须大于四

strides:每块patch区域之间中心点之间的距离，必须是: [1, stride_rows, stride_cols, 1].

rates:在原始图像的一块patch中，隔多少像素点，取一个有效像素点，必须是[1, rate_rows, rate_cols, 1]

padding:有两个取值，“VALID”或者“SAME”，“VALID”表示所取的patch区域必须完全包含在原始图像中."SAME"表示

可以取超出原始图像的部分，这一部分进行0填充。

tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 5, 5, 1], 
    rates=[1, 1, 1, 1], padding='VALID')

'''

# input an image, outputa for model.predict(x_valid)


if __name__ == '__main__':

    img_file = '/tmp4/img1.png'
    img1 = cv2.imread(img_file) # h:512, w:768

    img_height, img_width = img1.shape[:-1]
    img_empty = np.zeros((512, 768, 3))

    patch_h = 32
    patch_w = 64

    list1 = extract_patch_non_overlap(img_file, patch_h=patch_h, patch_w=patch_w)

    for index, img11 in enumerate(list1):
        filename = "/tmp4/{}.jpg".format(index)
        cv2.imwrite(filename, img11)

    # from top to bottom
    row_num = img_height // patch_h
    col_num = img_width // patch_w

    for index, patch1 in enumerate(list1):
        col = index // row_num
        row = index % row_num

        img_empty[row * patch_h:(row+1) * patch_h,
            col * patch_w:(col+1)*patch_w, :] =patch1


    cv2.imwrite('a.png', img_empty)

    print('OK')

