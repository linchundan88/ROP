import os

from LIBS.DataPreprocess.my_compute_digest import CalcSha1
from LIBS.DB.db_helper_conn import get_db_conn

db = get_db_conn()
cursor = db.cursor()

dir1 = '/media/ubuntu/data2/tmp5/2020_3_13_results/posterior/0'
i = 0
for dir_path, subpaths, files in os.walk(dir1, False):
    for f in files:
        image_file_source = os.path.join(dir_path, f)
        file_dir, filename = os.path.split(image_file_source)
        file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue

        sha1 = CalcSha1(image_file_source)

        sql_update = "update tb_multi_labels set posterior=1 where SHA1=%s "

        record_num = cursor.execute(sql_update, (sha1,))
        if record_num == 0:
            print(image_file_source)
        # print('update record:', record_num)

        i += 1
        if i % 20 == 0:
            db.commit()

db.commit()
db.close()

print('OK')