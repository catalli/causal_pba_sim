import numpy as np
import csv
import pbaclasses
import sys
import os
import math
import gc

print(sys.argv)

def randombinstring(length):
    string = ""
    for i in range(length):
        bit = np.random.binomial(size=None,n=1,p=0.5)
        string+=str(bit)
    return string

def hammstring(binstring1, binstring2):
    hamm = abs(len(binstring1)-len(binstring2))
    for j in range(min(len(binstring1),len(binstring2))):
        if binstring1[j] != binstring2[j]:
            hamm+=1
    return hamm

if len(sys.argv) == 1:
    msg_len = 2048

    dataname = "treestream"

    numinstances= 1000

    maxnumsteps = msg_len

    percentcheck = 0.95

    p = 0.757

    delta = 0.1

    deltapowers = [1,1.3,2,3,-1]

    k=0
    better_prior=False
    
datapath=''.join([os.path.dirname(os.path.realpath(__file__)),"/../data/",
        dataname, "%p", str(p), "%d",str(deltapowers),"%k", str(k), "%betterprior",
        str(better_prior), "%checkprop", str(percentcheck), "%ni",
        str(numinstances), ".csv"])

qvals = []

chancap = pbaclasses.bsc_capacity(p)

for power in deltapowers:
    if power == -1:
        qvals.append(chancap)
    else:
        qvals.append(chancap-delta**power)

datafile = open(datapath,'w',newline='')

datawriter = csv.writer(datafile, delimiter=',')


for q in qvals:
    distcounter=0
    for j in range(numinstances):
        true_msg = randombinstring(msg_len)
        distrib = pbaclasses.causal_pba_tree(p=p,true_message=true_msg, q=q,k=k,better_prior=better_prior,streaming=True)
        gc.collect()
        distcounter+=1
        for step in range(maxnumsteps):
            orac_val = distrib.oracle()
            g0mult = distrib.get_g0mult(orac_val) 
            g1mult = distrib.get_g1mult(orac_val) 
            distrib.update_posterior(g0mult, g1mult) 
            distrib.update_partition()
            m = int(float(max(len(distrib.accessed_msg),len(distrib.mode)))*percentcheck)
            distance = 1
            if distrib.accessed_msg[:m] == distrib.mode[:m]:
                distance = 0
            datarow = [str(distrib.stepno),str(distance),str(percentcheck),str(q),str(p),str(k),str(int(better_prior))]
            datawriter.writerow(datarow)
            logmsg = "".join(["Distrib: ", str(distcounter), "\tStep: ",
                str(distrib.stepno), "\tMessage length: ",
                str(len(distrib.accessed_msg)),"\tMode length: ",
                str(len(distrib.mode)),"\tDifference in checked bits: ", str(distance),
                " q_val: ", str(q)]) 
            print(logmsg)
            distrib.update_accessed_msg()
            distrib.stepno+=1

datafile.close()
