import numpy as np
import csv
import pbaclasses
import sys
import os
import math


if len(sys.argv) == 1:
    dataname = "outputmean"

    numinstances=100

    numsteps = 500

    p = 0.9

datapath=''.join([os.path.dirname(os.path.realpath(__file__)),"/../data/",
        dataname, "%p", str(p), "%tpDistributed", "%ni",
        str(numinstances), "%ns", str(numsteps), ".csv"])

experiments = []

for i in range(numinstances):
    truepoint = np.random.uniform()
    distrib = pbaclasses.pba_pdf(p,truepoint)
    experiments.append(distrib)

datafile = open(datapath,'w',newline='')

datawriter = csv.writer(datafile, delimiter=',')

for step in range(numsteps):
    datarow = [str(step)]
    l1vals = []
    l2vals = []
    for distrib in experiments:
        gamma = distrib.gamma(distrib.mean)
        l1vals.append(distrib.get_diff(distrib.mean))
        l2vals.append((distrib.get_diff(distrib.mean))**2)
        orac_val = distrib.oracle(distrib.mean)
        if orac_val == 1:
            distrib.add_breakpoint(distrib.mean,(gamma**(-1))*(1.0-distrib.p),
                    (gamma**(-1))*distrib.p)
        else:
            distrib.add_breakpoint(distrib.mean,((1.0-gamma)**(-1))*distrib.p,
                    ((1.0-gamma)**(-1))*(1.0-distrib.p))
        distrib.refresh_mean()
    l1min = min(l1vals)
    l2min = min(l2vals)
    underflows=0
    for i in range(len(l1vals)):
        if l1vals[i] >= 1.0:
            l1vals[i] = l1min
            l2vals[i] = l2min
            underflows+=1
    expl1 = math.fsum(l1vals)/numinstances
    expl2 = math.fsum(l2vals)/numinstances
    datarow.append(str(expl1))
    datarow.append(str(expl2))
    datarow.append(str(expl2-expl1**2))
    datarow.append(str(underflows))
    logmsg = ["step: ",str(step),"\t","E(L1): ",str(expl1),
            "\t", "E(L2): ",str(expl2),"\t","stddev: ",
            str(expl2-expl1**2),"\t","underflows: ", str(underflows)]
    print(''.join(logmsg))
    datawriter.writerow(datarow)

datafile.close()
