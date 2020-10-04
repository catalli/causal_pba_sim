import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys
import os
from pbaclasses import avg_rate, bsc_capacity
sns.set(style="darkgrid")

if len(sys.argv) == 1:
    dataname = "outputtree"
    p_val = 0.947
    q_val = 0.95

datapath=''.join([os.path.dirname(os.path.realpath(__file__)),"/../data/"])

datafiles = []

for filename in os.listdir(datapath):
    if filename.endswith(".csv"):
        datafiles.append(os.path.join(datapath, filename))

datadict = {'-Log10(epsilon)':[], 'Average step at decode':[], 'Prob. of wrong decode':[], 'Normalized hamm. from target':[], 'Message length':[], 'k':[], 'Better prior updating':[]}

for filepath in datafiles:
    csvfile = open(filepath, newline='')
    datareader = csv.reader(csvfile, delimiter=",")
    if (dataname not in filepath) or ("p"+str(p_val) not in filepath) or ("q"+str(q_val) not in filepath):
        csvfile.close()
        continue
    msg_len = int((os.path.basename(filepath).split("%")[-3])[7:])
    if msg_len < 8:
        csvfile.close()
        continue
    for row in datareader:
        if int(row[4]) == 1:
            datadict['-Log10(epsilon)'].append(float(row[0]))
            datadict['Message length'].append(msg_len)
            datadict['Average step at decode'].append(float(row[1]))
            datadict['Prob. of wrong decode'].append(float(row[2]))
            datadict['Normalized hamm. from target'].append(float(row[3])/msg_len)
            datadict['k'].append(int(row[6]))
            if bool(int(row[7])):
                datadict['Better prior updating'].append("Yes")
            else:
                datadict['Better prior updating'].append("No")
    csvfile.close()

plot_data = pd.DataFrame(data=datadict)
avgdict = {'-Log10(epsilon)':[], 'Message length':[], 'Rate':[], 'Error exponent':[], 'Value type':[], 'k':[], 'Better prior updating':[]}
msglens = plot_data['Message length'].unique()
epsilons = plot_data['-Log10(epsilon)'].unique()
ks = plot_data['k'].unique()
betterpriors = plot_data['Better prior updating'].unique()
for msg_len in msglens:
    for epsilon in epsilons:
        for k in ks:
            for betterprior in betterpriors:
                avgdict['-Log10(epsilon)'].append(epsilon)
                avgdict['Message length'].append(msg_len)
                avgtau = plot_data.loc[plot_data['-Log10(epsilon)']==epsilon].loc[plot_data['Message length']==msg_len].loc[plot_data['k']==k].loc[plot_data['Better prior updating']==betterprior, 'Average step at decode'].mean()
                avgdict['Rate'].append(float(msg_len)/float(avgtau))
                avgdict['Error exponent'].append(float(epsilon)/float(avgtau))
                avgdict['Value type'].append("numeric")
                avgdict['k'].append(k)
                avgdict['Better prior updating'].append(betterprior)
    avgdict['-Log10(epsilon)'].append(np.max(epsilons)*1.1)
    avgdict['Message length'].append(msg_len)
    try:
        avgdict['Rate'].append(avg_rate(p_val,q_val,msg_len))
    except:
        avgdict['Rate'].append(bsc_capacity(p_val))
    avgdict['Error exponent'].append(np.nan)
    avgdict['Value type'].append("asymptotic upper bound")
    avgdict['k'].append(-1)
    avgdict['Better prior updating'].append("Yes")

avg_data = pd.DataFrame(data=avgdict)
        

axtau = sns.relplot(x="-Log10(epsilon)",y="Average step at decode", hue="Message length", style="k", kind="line",data=plot_data);
axwdecode = sns.relplot(x="-Log10(epsilon)",y="Prob. of wrong decode", hue="Message length", style="k", kind="line", data=plot_data);
axhamm = sns.relplot(x="-Log10(epsilon)",y="Normalized hamm. from target", hue="Message length",style="k", kind="line", data=plot_data);
sns.relplot(x="Message length",y="Rate", hue="-Log10(epsilon)", style="k", kind="line",data=avg_data);
sns.relplot(x="-Log10(epsilon)",y="Error exponent", hue="Message length", style="k", kind="line",data=avg_data);


axwdecode.set(yscale="log")
axhamm.set(yscale="log")

plt.show()
