import csv
import numpy as np
import matplotlib.pyplot as plt
txt_file = r"outdir.out.Vm"
csv_file = r"my.csv"


in_txt = csv.reader(open(txt_file, "rb"), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wb'))

out_csv.writerows(in_txt)



