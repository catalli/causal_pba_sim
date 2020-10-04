import numpy as np
import csv
import pbaclasses
import sys
import os
import math

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
    msg_len = 8

    dataname = "outputtree"

    numinstances= 10000

    maxnumsteps = 100000

    epsilonbase = 0.1

    epsilonpowers = [1,2,3,4,5 ]

    p = 0.757
    q = 0.5
    k=0
    better_prior = False

datapath=''.join([os.path.dirname(os.path.realpath(__file__)),"/../data/",
        dataname, "%p", str(p), "%q", str(q), "%k", str(k), "%betterprior",
        str(better_prior),
        "%msg_len", str(msg_len),
        "%tp", "Distributed", "%ni",
        str(numinstances), ".csv"])

errthreshes = []

for power in epsilonpowers:
    errthreshes.append((1.0-epsilonbase**power))

datafile = open(datapath,'w',newline='')

datawriter = csv.writer(datafile, delimiter=',')

for experiment in range(numinstances):
    powersreached = []
    true_msg=randombinstring(msg_len)
    distrib = pbaclasses.causal_pba_tree(p=p,true_message=true_msg,q=q,k=k, better_prior=better_prior)
    for step in range(maxnumsteps):
        if len(powersreached) == len(epsilonpowers):
            break
        orac_val = distrib.oracle()
        g0mult = distrib.get_g0mult(orac_val) 
        g1mult = distrib.get_g1mult(orac_val) 
        distrib.update_posterior(g0mult, g1mult)
        distrib.update_partition()
        for power in range(len(epsilonpowers)):
            if (epsilonpowers[power] not in powersreached) and ((distrib.modeprob > errthreshes[power]) or step == maxnumsteps-2) and distrib.stepno >= msg_len-1:
                wconv = 0
                hamm = 0
                converged = 1
                if step == maxnumsteps -2:
                    converged = 0
                if distrib.mode != distrib.accessed_msg:
                    wconv = 1
                    hamm = hammstring(distrib.mode,distrib.accessed_msg)
                datarow = [str(epsilonpowers[power]),str(distrib.stepno),str(wconv),str(hamm),str(converged),str(msg_len),str(k),str(int(better_prior))]
                datawriter.writerow(datarow)
                logmsg = "".join(["Distrib: ", str(experiment),
                    "\tMessage len: ", str(len(distrib.accessed_msg)), "\tErrthresh: ", str(errthreshes[power]), 
                    "\tDecodestep: ", str(distrib.stepno), "\tMode len: ", str(len(distrib.mode))]) 
                print(logmsg)
                powersreached.append(epsilonpowers[power])
        distrib.update_accessed_msg()
        distrib.stepno+=1
datafile.close()
