import os
import pandas as pd
from LIBS.DB.db_helper_conn import get_db_conn
from LIBS.DataPreprocess.my_compute_digest import CalcSha1
import shutil

def write_csv_to_db():
    filename_csv = os.path.join(os.path.abspath('..'),
                  'datafiles',  'dataset6', 'Plus_step_two.csv')

    db = get_db_conn()
    cursor = db.cursor()

    df = pd.read_csv(filename_csv)
    for _, row in df.iterrows():
        image_file = row['images']
        image_label = int(row['labels'])

        # blood_vessel_seg_result_2020_4_27 blood_vessel_seg_result1  blood_vessel_seg_result

        image_file = image_file.replace('/blood_vessel_seg_result_2020_4_27/', '/original/')
        image_file = image_file.replace('/blood_vessel_seg_result1/', '/original/')
        image_file = image_file.replace('/blood_vessel_seg_result/', '/original/')

        # print(image_file)
        sha1 = CalcSha1(image_file)

        sql_update = 'update tb_multi_labels set plus3=%s where sha1=%s'
        cursor.execute(sql_update, (image_label, sha1))
        db.commit()

        rowcount = cursor.rowcount
        if rowcount == 0:
            print(image_file)

            (filepath, tempfilename) = os.path.split(image_file)
            # (filename, extension) = os.path.splitext(tempfilename)

            tempfilename_new = sha1 + '#' + tempfilename

            image_file_dest = os.path.join('/tmp5/2020_5_21_plus', str(image_label), tempfilename_new)
            if not os.path.exists(os.path.dirname(image_file_dest)):
                os.makedirs(os.path.dirname(image_file_dest))

            shutil.copy(image_file, image_file_dest)

            # sql_insert = 'insert into tb_multi_labels(SHA1,filename, plus3) values(%s, %s, %s)'
            # cursor.execute(sql_insert, (sha1, tempfilename_new, image_label))
            # db.commit()
        else:
            pass


def export_db_dir(source_dir, dest_dir):
    db = get_db_conn()
    cursor = db.cursor()

    sql = 'select SHA1,filename, plus3 from tb_multi_labels where plus3 is not null and uncertain is null and other_diseases is null '
    cursor.execute(sql)
    results = cursor.fetchall()

    for rs in results:
        sha1 = rs[0]
        filename = rs[1]

        filename_original = os.path.join(source_dir,
                    sha1 + '#' + filename)

        if os.path.exists(filename_original):
            print(filename_original)
            filename_dest = os.path.join(dest_dir, str(rs[2]),
                                         sha1 + '#' + filename)

            if not os.path.exists(os.path.dirname(filename_dest)):
                os.makedirs(os.path.dirname(filename_dest))
            shutil.copy(filename_original, filename_dest)
        else:
            print(filename_original, 'not found!')

# source_dir = '/media/ubuntu/data1/ROP_dataset1/original'
# dest_dir = '/media/ubuntu/data1/ROP_dataset_plus_2020_5_19/original'
# export_db_dir(source_dir, dest_dir)

write_csv_to_db()

print('OK')