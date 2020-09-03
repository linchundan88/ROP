import os
import MySQLdb
import json

def get_db_conn():
    json_file = os.path.join(os.path.dirname(__file__), 'db_config.json')
    assert os.path.exists(json_file), 'db_config.json does not exist.'

    with open(json_file, 'r') as load_f:
        data = json.load(load_f)
        db = MySQLdb.connect(data['host'], data['username'], data['password'],
                         data['database'], use_unicode=True, charset='utf8')

        return db

