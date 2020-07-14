import pandas as pd
import cv2
from LIBS.ImgPreprocess.my_image_helper import image_to_square

# opencv can not read .gif because of license problem.
# import imageio

# dir_path = '/media/ubuntu/data1/公开数据集/BloodVesselsSegment/original/DRIVE/'
# for dir_path, subpaths, files in os.walk(dir_path, False):
#     for f in files:
#         img_file_source = os.path.join(dir_path, f)
#
#         filename, file_extension = os.path.splitext(img_file_source)
#
#         if file_extension == '.gif':
#             img_file_dest = img_file_source.replace('.gif', '.png')
#
#             img1 = imageio.imread(img_file_source)
#             print(img_file_dest)
#             imageio.imwrite(img_file_dest, img1)



filename_csv = 'BloodVessel384.csv'
df = pd.read_csv(filename_csv)  # panda dataframe

count = len(df.index)
for i in range(count):
    img_file = df.at[i, 'images']
    img_mask = df.at[i, 'masks']

    if not 'HRF' in img_file:
        continue

    print(img_file)

    img1 = cv2.imread(img_file)
    img2 = image_to_square(img1)
    cv2.imwrite(img_file, img2)

    img3 = cv2.imread(img_mask)
    img4 = image_to_square(img3)
    cv2.imwrite(img_mask, img4)

print('OK')