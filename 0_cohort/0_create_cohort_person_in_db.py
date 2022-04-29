# In[ ]:
# ** import package **
import os
import sys
import json
import pathlib
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
from datetime import timedelta
from _utils.customlogger import customlogger as CL

pd.set_option('display.max_colwidth', -1)    #각 컬럼 width 최대로 
pd.set_option('display.max_rows', 50)        # display 50개 까지 

# In[ ]:
# ** loading config **
with open('./../{}'.format("config.json")) as file:
    cfg = json.load(file)

# In[ ]:
# ** loading info **
current_dir = pathlib.Path.cwd()
parent_dir = current_dir.parent
current_date = cfg["working_date"]
curr_file_name = os.path.splitext(os.path.basename(__file__))[0]

# In[ ]:
# **create Logger**
log = CL("custom_logger")
pathlib.Path.mkdir(pathlib.Path('{}/_log/'.format(parent_dir)), mode=0o777, parents=True, exist_ok=True)
log = log.create_logger(file_name="../_log/{}.log".format(curr_file_name), mode="a", level="DEBUG")    
log.debug('start {}'.format(curr_file_name))

# In[ ]:
# ** connection DataBase **
if (cfg["dbms"]=="postgresql"):
    db_cfg = cfg["postgresql"]
    import psycopg2 as pg
    conn = pg.connect(host=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], dbname=db_cfg['@database']) 
    log.debug("postgresql connect")
    
elif (cfg["dbms"]=="mssql"):
    db_cfg = cfg["mssql"]
    import pymssql
    conn= pymssql.connect(server=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], database=db_cfg['@database'], as_dict=False)
    log.debug("mssql connect")
    
else:
    log.warning("set config.json - sql - dbms : mssql or postgresql")

# In[ ]:
def renderTranslateQuerySql(sql_query, dict):
    '''
    ----- Replace text ------
    "@cdm_database_schema",
    "@target_database_schema",
    "@target_cohort_table",
    "@vocabulary_database_schema",
    "@target_cohort_id"
    ----- Replace text ------
    '''
    for key, value in dict.items():
        sql_query = sql_query.replace(key, value)
    return sql_query

# In[ ]:
try:
    with conn.cursor() as cursor:
        for file_path in cfg['translatequerysql0']:
            param_dict = cfg['translatequerysql0'][file_path]
            sql_file_path = file_path.replace("{dbms}", cfg["dbms"])
            print(sql_file_path)
            print(param_dict)
            fd = open(sql_file_path, 'r')
            sql_query = fd.read()
            sql_query = renderTranslateQuerySql(sql_query, param_dict)
            fd.close()
            cursor.execute(sql_query)
        conn.commit()
except :
    traceback.print_exc()
    log.error(traceback.format_exc())

# In[ ]:
try:
    with conn.cursor() as cursor:
        for file_path in cfg['translatequerysql1']:
            param_dict = cfg['translatequerysql1'][file_path]
            for drug in cfg['drug'].keys():
                sql_param_dict = param_dict.copy()
                for param_key, param_value in sql_param_dict.items():
                    temp_param_value = param_value
                    temp_param_value = temp_param_value.replace("{drug}", drug)
                    temp_param_value = temp_param_value.replace("{drug_target_cohort_id}", cfg['drug'][drug]["drug_target_cohort_id"])
                    sql_param_dict[param_key] = temp_param_value
                sql_file_path = file_path.replace("{dbms}", cfg["dbms"]).replace("{drug}", drug)
                print(sql_file_path)
                print(sql_param_dict)
                fd = open(sql_file_path, 'r')
                sql_query = fd.read()
                sql_query = renderTranslateQuerySql(sql_query, sql_param_dict)
                fd.close()
                cursor.execute(sql_query)
        conn.commit()
except :
    traceback.print_exc()
    log.error(traceback.format_exc())

# In[ ]:
try:
    with conn.cursor() as cursor:
        for file_path in cfg['translatequerysql2']:
            param_dict = cfg['translatequerysql2'][file_path]
            for meas in cfg['meas'].keys():
                sql_param_dict = param_dict.copy()
                for param_key, param_value in sql_param_dict.items():
                    temp_param_value = param_value
                    temp_param_value = temp_param_value.replace("{meas}", meas)
                    temp_param_value = temp_param_value.replace("{@meas_concept_id}", cfg['meas'][meas]["@meas_concept_id"])
                    sql_param_dict[param_key] = temp_param_value
                sql_file_path = file_path.replace("{dbms}", cfg["dbms"])
                print(sql_file_path)
                print(sql_param_dict)
                fd = open(sql_file_path, 'r')
                sql_query = fd.read()
                sql_query = renderTranslateQuerySql(sql_query, sql_param_dict)
                fd.close()
                cursor.execute(sql_query)
        conn.commit()
except :
    traceback.print_exc()
    log.error(traceback.format_exc())
    
# In[ ]:
conn.close()
# %%
