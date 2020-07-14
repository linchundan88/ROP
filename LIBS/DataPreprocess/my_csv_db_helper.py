import csv
import os

from LIBS.DB.db_helper_conn import get_db_conn


def export_csv_from_db(base_dir, sql, file_csv):

    if not os.path.exists(os.path.dirname(file_csv)):
        os.makedirs(os.path.dirname(file_csv))

    db_con = get_db_conn()
    cursor = db_con.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()

    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for rs in results:
            sha1 = rs[0]
            filename = rs[1]
            img_file = os.path.join(base_dir, sha1 + '#' + filename)
            if not os.path.exists(img_file):
                # raise Exception('file not found,', img_file)
                print('file not found,', img_file)
                continue

            class_labels = rs[2]

            csv_writer.writerow([img_file, class_labels])

    db_con.close()