import os, csv, cv2

dir_path = '/media/ubuntu/data1/公开数据集/BloodVesselsSegment/original/Stare'


# for dir_path, subpaths, files in os.walk(dir_path, False):
#     for f in files:
#         img_file_source = os.path.join(dir_path, f)
#
#         filename, file_extension = os.path.splitext(img_file_source)
#
#         if file_extension.upper() == '.PPM':
#             img_file_dest = img_file_source.replace('.ppm', '.png')
#             img1 = cv2.imread(img_file_source)
#             cv2.imwrite(img_file_dest, img1)
#
# exit(0)

file_csv = 'Stare.csv'

if os.path.exists(file_csv):
    os.remove(file_csv)

with open(file_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'masks'])

    dir_path = '/media/ubuntu/data1/公开数据集/BloodVesselsSegment/original/Stare/stare-images'

    for dir_path, subpaths, files in os.walk(dir_path, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF', '.PPF']:
                print('file ext name:', f)
                continue

            img_file_mask = img_file_source.replace('/stare-images/', '/labels-ah/')
            img_file_mask = img_file_mask.replace('.png', '.ah.png')

            if os.path.exists(img_file_mask):
                csv_writer.writerow([img_file_source, img_file_mask])
