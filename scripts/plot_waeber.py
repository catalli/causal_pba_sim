import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys
import os
sns.set(style="darkgrid")

steplimit = 250

stepmin = 0


if len(sys.argv) == 1:
    dataname = "output"

datapath=''.join([os.path.dirname(os.path.realpath(__file__)),"/../data/"])

datafiles = []

for filename in os.listdir(datapath):
    if filename.endswith(".csv"):
        datafiles.append(os.path.join(datapath, filename))

datadict = {'Step':[], 'ln(Mean error)':[], 'p':[], 'Method':[], 'Error type':[]}

for filepath in datafiles:
    csvfile = open(filepath, newline='')
    datareader = csv.reader(csvfile, delimiter=",")
    pvalue=filepath.split('%')[-4]
    if pvalue != "p0.75" or ("ni1000" not in os.path.basename(filepath)):
        csvfile.close()
        continue
    method = "PBA"
    if "mean" in  os.path.basename(filepath):
        method = "Mean-Querying"
    for row in datareader:
        if int(row[0]) <= steplimit and int(row[0]) >= stepmin and float(row[1]) < 1.0:
            datadict['Step'].append(float(row[0]))
            datadict['Step'].append(float(row[0]))
            datadict['ln(Mean error)'].append(np.log(float(row[1])))
            datadict['ln(Mean error)'].append(np.log(float(row[2])))
            datadict['Error type'].append("L1")
            datadict['Error type'].append("L2")
            datadict['Method'].append(method)
            datadict['Method'].append(method)
            datadict['p'].append(float(pvalue[1:]))
            datadict['p'].append(float(pvalue[1:]))
    csvfile.close()

plot_data = pd.DataFrame(data=datadict)

sns.relplot(x="Step",y="ln(Mean error)", hue="Method", style="Error type", hue_order=["PBA","Mean-Querying"],
        data=plot_data);
plt.show()
