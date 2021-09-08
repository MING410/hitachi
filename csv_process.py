import pandas
import numpy as np
import csv
import os
import re
from openpyxl import load_workbook

base_dir='/Users/momo/Desktop/KIMORE/'
for dirpath, dirnames, filenames in os.walk(base_dir, followlinks=True):
    # Es1
    pattern = re.compile(r'^[^\.].*\.(csv|xlsx)$')
    for filename in filenames:
        if not pattern.match(filename): continue
        if "Es1" not in dirpath:
            continue
        if 'JointPosition' in filename:
            position_path = os.path.join(dirpath, filename)
            if os.path.exists(position_path):
                # read data
                # f = open(position_path, 'r', encoding='utf-8')
                # with open(position_path, 'w') as csv_file:
                #     writer = csv.writer(csv_file, dialect='excel')
                f = open(position_path, encoding="gbk")
                content = f.read()
                f.close()
                t = content.replace("，", ",")
                #position_newpath=position_path.replace('JointPosition','JointPosition_new')
                with open(position_newpath, "w", encoding='gbk') as f1:
                    f1.write(t)