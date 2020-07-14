import cv2

img1 = cv2.imread('rop1.jpg')

img2 = img1[:,:,1]
cv2.imwrite('rop1_green.jpg', img2)


img2 = img1[:,:,0]
cv2.imwrite('rop1_blue.jpg', img2)

img2 = img1[:,:,2]
cv2.imwrite('rop1_red.jpg', img2)
