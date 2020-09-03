import os

from LIBS.DB.db_helper_conn import get_db_conn


db_con = get_db_conn()
cursor = db_con.cursor()
sql = 'select ID,SHA1,filename,filepath, gradable3, hemorrhage3, stage3, posterior  from tb_multi_labels'
cursor.execute(sql)
results = cursor.fetchall()

for rs in results:
    ID = rs[0]
    SHA1 = rs[1]
    print(SHA1)

print('OK')