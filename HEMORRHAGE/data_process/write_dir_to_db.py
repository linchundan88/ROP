import os
import shutil
from LIBS.DataPreprocess.my_compute_digest import CalcSha1
from LIBS.DB.db_helper_conn import get_db_conn


def write_sha1_db():
    db = get_db_conn()
    cursor = db.cursor()

    dir1 = '/media/ubuntu/data1/ROP_dataset/'
    dir2 = '/media/ubuntu/data1/ROP_dataset_2/'

    for dir_path, subpaths, files in os.walk(dir1, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)
            if '/original/' not in image_file_source:
                continue

            file_dir, filename = os.path.split(image_file_source)

            file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            print(image_file_source)

            sha1 = CalcSha1(image_file_source)

            image_file_dest = os.path.join(dir2, sha1 + '#' + filename)
            if not os.path.exists(os.path.dirname(image_file_dest)):
                os.makedirs(os.path.dirname(image_file_dest))
            shutil.copy(image_file_source, image_file_dest)

            sql_delete = 'delete from tb_multi_labels where SHA1=%s'
            cursor.execute(sql_delete, (sha1,))

            sql_insert = 'insert into tb_multi_labels(SHA1,filename) values(%s, %s)'
            cursor.execute(sql_insert, (sha1, filename))

            db.commit()

    db.close()

def write_labels_to_db(dir1, dict_mapping, db_field_name='stage'):

    db = get_db_conn()
    cursor = db.cursor()

    i = 0
    for dir_path, subpaths, files in os.walk(dir1, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)
            file_dir, filename = os.path.split(image_file_source)
            file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            sha1 = CalcSha1(image_file_source)
            # print(image_file_source)

            is_found = False
            for key in dict_mapping:
                if '/' + key + '/' in image_file_source:
                    label = dict_mapping[key]
                    is_found = True

            if is_found:
                print('label:', label, image_file_source)
            else:
                raise Exception("Invalid label!", image_file_source)

            sql_update = "update tb_multi_labels set {0}=%s where SHA1=%s and ({0} is null or {0} <>%s)".format(db_field_name)

            record_num = cursor.execute(sql_update, (label, sha1, label))
            if record_num == 0:
                print(image_file_source)
            # print('update record:', record_num)

            i += 1
            if i % 20 == 0:
                db.commit()

    db.commit()
    db.close()


dict_mapping_hemorrhage = {'0': 0, '对照组': 0, '1': 1, '出血': 1}
dir1 ='/media/ubuntu/data1/ROP_dataset/Hemorrhage/original'
# write_labels_to_db(dir1, dict_mapping=dict_mapping_hemorrhage, db_field_name='hemorrhage')

db = get_db_conn()
cursor = db.cursor()
sql_update = 'update tb_multi_labels set hemorrhage1=hemorrhage'
cursor.execute(sql_update)
db.commit()
db.close()

dir1 = '/tmp5/result_20200308_confusion_matrix_postmodification/hemorrhage'
write_labels_to_db(dir1, dict_mapping=dict_mapping_hemorrhage, db_field_name='hemorrhage1')



print('OK')