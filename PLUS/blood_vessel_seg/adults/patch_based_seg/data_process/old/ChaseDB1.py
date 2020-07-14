import os, csv


file_csv = 'ChaseDB1.csv'

if os.path.exists(file_csv):
    os.remove(file_csv)

with open(file_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'masks'])

    dir_path = '/media/ubuntu/data1/公开数据集/BloodVesselsSegment/original/ChaseDB1'

    for dir_path, subpaths, files in os.walk(dir_path, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            if file_extension == '.jpg':
                img_file_mask = img_file_source.replace('.jpg', '_1stHO.png')

                if os.path.exists(img_file_mask):
                    csv_writer.writerow([img_file_source, img_file_mask])
