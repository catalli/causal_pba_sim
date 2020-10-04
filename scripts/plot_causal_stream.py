import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys
import os
sns.set(style="darkgrid")

if len(sys.argv) == 1:
    dataname = "treestream"
    maxstepnum = 2048
    p_val = 0.757
    perc_check_val=0.95

datapath=''.join([os.path.dirname(os.path.realpath(__file__)),"/../data/"])

datafiles = []

for filename in os.listdir(datapath):
    if filename.endswith(".csv"):
        datafiles.append(os.path.join(datapath, filename))

datadict = {'Step number':[], 'Probability of instability':[], 'Proportion of bits checked':[], 'q':[], 'p':[], 'k':[], 'Better prior updating':[]}

for filepath in datafiles:
    csvfile = open(filepath, newline='')
    datareader = csv.reader(csvfile, delimiter=",")
    if (dataname not in filepath):
        csvfile.close()
        continue
    for row in datareader:
        if (float(row[2]) == perc_check_val) and (float(row[4]) ==p_val) and (int(row[0]) < maxstepnum):
            datadict['Step number'].append(float(row[0]))
            datadict['Probability of instability'].append(float(row[1]))
            datadict['Proportion of bits checked'].append(float(row[2]))
            datadict['q'].append(float(row[3]))
            datadict['p'].append(float(row[4]))
            datadict['k'].append(int(row[5]))
            if bool(int(row[6])):
                datadict['Better prior updating'].append("Yes")
            else:
                datadict['Better prior updating'].append("No")
    csvfile.close()
plot_data = pd.DataFrame(data=datadict)

axhamm = sns.relplot(x="Step number",y="Probability of instability", hue="q",
         style="Better prior updating",
         data=plot_data, kind="line");
axhamm.set(yscale="log")

plt.show()
