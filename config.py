# encoding: utf-8
"""
@author: guozhenyu 
@contact: guozhenyu@pku.edu.cn

@version: 1.0
@file: config.py
@time: 2023/9/10 6:05 PM
"""

host = '127.0.0.1'
user = 'root'
# pwd = '2023Tianzhi_Zhenyu'
pwd='root'
port = 3306
dbnam = 'liyi'
conn = f'mysql+pymysql://{user}:{pwd}@{host}:{port}/{dbnam}'
