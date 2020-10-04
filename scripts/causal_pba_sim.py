"""
causal_pba_sim.py

This package models a causal and
non-causal setting for error-correction
over a communications channel with feedback.

The non-causal setting models the message as a point
on [0,1) and utilizes Waeber's Probabilistic Bisection
Algorithm (PBA). The causal setting assumes a new bit
of the true message arrives with some fixed probability
at every step. The algorithms modeled for the causal
setting are a sorting-based approximation of the partition
problem over the probability mass function on messages
and a tree-based approximation.
"""

import numpy as np
import math

class waeber_pba:
    """Implements Waeber's setting of the PBA on [0,1).
    Stores a piecewise-constant probability density function
    for a target point and implements methods to query an
    oracle and update this PDF via Bayes' rule.
    """
    def __init__(self,p=0.6,truepoint=0.6):
        self.breakpoints=np.array([[1.0,1.0]])
        """A NumPy 2xn ndarray
        storing the piecewise-constant PDF. The first
        element in each row is the open end of the interval
        (starting from 0 if no other starting point is defined)
        over which the PDF has the value given in the second
        element. Initialized with a single element ``[1.0,1.0]`` to
        model a uniform probability distribution on [0,1).
        """ 
        self.median=0.5
        """Float storing the current median of the PDF."""
        self.mean=0.5
        """Float storing the current mean of the PDF."""
        self.p=p
        """Probability in range (0,1) that the oracle will answer
        truthfully.
        """
        self.truepoint=truepoint
        """Target point representing the message the decoder is trying
        to locate.
        """
    def evaluate(self, x):
        """Evaluates the PDF at a point.
        
        :param x: The point to evaluate the PDF at.
        
        :returns: The value of the PDF at ``x``.
        """
        if x < 0.0 or x >= 1.0:
            return 0.0
        for val in self.breakpoints:
            if val[0] > x:
                return val[1]
    def oracle(self, x, correct_prob=None):
        """Query a noisy oracle to determine if the true
        point is greater or less than a queried point.

        :param x: The point being queried.
        :param correct_prob: The probability that the oracle
            will answer truthfully. ``p`` by default.

        :returns: With probability ``correct_prob``,
            ``1`` if ``truepoint`` is greater than or
            equal to ``x``, ``-1`` otherwise. With probability
            ``1-correct_prob``, returns flipped values.
        """
        if correct_prob is None:
            correct_prob = self.p
        true_eval=1
        flip = np.random.binomial(size=None,n=1,p=correct_prob)
        if self.truepoint < x:
            true_eval=-1
        return true_eval*((2*flip)-1)
    def cdf(self, x):
        """Cumulative distribution function of stored PDF.

        :param x: The point at which to evaluate the CDF.

        :returns: CDF of the PDF at point ``x``.
        """
        prevx = 0.0
        currenty = 0.0
        sumlist = []
        for point in self.breakpoints:
            currenty = point[1]
            if x < point[0]:
                break
            sumlist.append((point[0]-prevx)*currenty)
            prevx = point[0]
        sumlist.append((x-prevx)*currenty)
        return math.fsum(sumlist)
    def gamma(self, x):
        """Gamma function used in Bayesian update.

        :param x: Point at which to evaluate function.
        
        :returns: (1-``cdf(x)``)p+``cdf(x)``(1-p).
        """
        F = self.cdf(x)
        return ((1.0-F)*self.p)+(F*(1.0-self.p))
    def refresh_mean(self):
        """Re-calculates mean using analytically
        calculated integrals on PDF. Stores updated
        value in ``mean``.
        """
        prevx = 0.0
        sumlist = []
        for point in self.breakpoints:
            integrandtop = point[0]**2
            integrandbottom = prevx**2
            sumlist.append(integrandtop*point[1]/2)
            sumlist.append(-integrandbottom*point[1]/2)
            prevx = point[0]
        self.mean = math.fsum(sumlist)
    def refresh_median(self):
        """Re-calculates median by analytically computing
        CDF in blocks until it reaches 0.5, then solves
        a linear equation to find the offset from the
        neares breakpoint. Stores updated value in
        ``median``.
        """
        prevx = 0.0
        cumint = 0.0
        currenty = 0.0
        sumlist = []
        for point in self.breakpoints:
            currenty = point[1]
            increment = [(currenty*(point[0]-prevx))]
            if math.fsum(sumlist+increment) >= 0.5:
                break
            else:
                sumlist.append(increment[0])
                prevx = point[0]
        targetoffset = 0.5-math.fsum(sumlist)
        xoffset = targetoffset/currenty
        self.median = prevx+xoffset
    def add_breakpoint(self, new_point, left_adjust=1.0, right_adjust=1.0):
        """Adds a new breakpoint to ``breakpoints``, and multiplies PDF by
        different values to its left and right.
        
        :param new_point: Point at which to add the new breakpoint.
        :param left_adjust: Value by which to multiply PDF to the left
            of ``new_point``.
        :param right_adjust: Value by which to multiply PDF to the right
            of ``new_point``.
        """
        overlapping=False
        for val in self.breakpoints:
            if val[0] == new_point:
                overlapping=True
                break
        if overlapping:
            new_breakpoints = np.zeros(self.breakpoints.shape)
            for i in range(self.breakpoints.shape[0]):
                new_breakpoints[i][0] = self.breakpoints[i][0]
                if self.breakpoints[i][0] <= new_point:
                    new_breakpoints[i][1] = self.breakpoints[i][1]*left_adjust
                else:
                    new_breakpoints[i][1]=self.breakpoints[i][1]*right_adjust
            self.breakpoints = new_breakpoints
        else:
            new_breakpoints = np.zeros((self.breakpoints.shape[0]+1,2))
            split_interval_index = 0
            for i in range(self.breakpoints.shape[0]):
                if new_point < self.breakpoints[i][0]:
                    split_interval_index = i
                    break
            for i in range(new_breakpoints.shape[0]):
                if i == split_interval_index:
                    new_breakpoints[i][0]=new_point
                    new_breakpoints[i][1]=self.breakpoints[i][1]*left_adjust
                elif i < split_interval_index:
                    new_breakpoints[i][0]=self.breakpoints[i][0]
                    new_breakpoints[i][1]=self.breakpoints[i][1]*left_adjust
                else:
                    new_breakpoints[i][0]=self.breakpoints[i-1][0]
                    new_breakpoints[i][1]=self.breakpoints[i-1][1]*right_adjust
            self.breakpoints=new_breakpoints
    def get_diff(self,x):
        """Get difference between ``truepoint`` and
        a given point.
        
        :param x: Point from which difference is measured.

        :returns: ``|x-truepoint|``.
        """
        return abs(x-self.truepoint)

class causal_setting:
    """Class providing framework for a causal setting
    with random bit arrivals. This includes the oracle's
    truth probability, the bit arrival probability, and a
    message source as well as the encoder's currently-held
    message.
    """
    def __init__(self, p=0.9, true_message="0", q=0.9, eta=0.9):
        self.p = p
        """Truth-probability of the BSC with feedback"""
        self.q = q
        """Bit arrival probability at each step"""
        self.msg = true_message
        """Full message up to maximu length that can
        possibly be received
        """
        self.n = len(true_message)
        """Length limit of message"""
        self.accessed_msg = true_message[0]
        """Message the decoder has currently accessed, initially
        first bit of ``true_msg``
        """
        self.g0prob = 0.5
        """Total probability mass in bin G0"""
        self.msgbin = 1
        """Bin that ``accessed_msg`` is currently in"""
        self.mode = "0"
        """Estimate of decoder"""
        self.eta = eta
        """Parameter eta for ternary-output channel"""
        self.stepno = 1
        """Number of steps the process is on"""
        if accessed_msg == "0":
            self.msgbin = 0
    def update_accessed_msg(self, addedchar="", update_prob=None):
        """Append the next bit from ``msg`` to ``accessed_msg`` with
        probability ``update_prob``.
        
        :param addedchar: If blank, set to next character in ``msg``
            if maximum length has not been reached. Otherwise an empty
            string.
        :param update_prob: Probability that update occurs. Defaults
            to ``q``.
        """
        if update_prob is None:
            update_prob = self.q
        if addedchar=="":
            if len(self.accessed_msg) < len(self.msg):
                addedchar = self.msg[len(self.accessed_msg)]
        updated = np.random.binomial(size=None, n=1, p=update_prob)
        if updated == 1:
            self.accessed_msg+=addedchar
    def bsc_oracle(self, true_prob=None):
        """With truth probability ``true_prob``, say if
        ``accessed_msg`` is in bin G0 or G1.

        :param true_prob: Probability that oracle tells the
            truth. Defaults to ``p``.

        :returns: ``0`` if ``accessed_msg`` is in G0, ``1``
            otherwise.
        """
        if true_prob is None:
            true_prob = self.p
        truth = np.random.binomial(size=None, n=1,p=true_prob)
        if truth ==1:
            return self.msgbin
        else:
            if self.msgbin == 0:
                return 1
            else:
                return 0
    def bayesian_multipliers_bsc(self, orac_val, p=None, g0prob=None):
        """Get multipliers for the bayesian update to be performed
        on G0 and G1 given a value from the BSC oracle.
        
        :param orac_val: Value given from oracle.
        :param p: Truth value of oracle. Defaults to ``p``.
        :g0prob: Total probability mass of G0. Defaults to ``g0prob``.

        :returns: A list ``{"G0":g0mult, "G1":g1mult}``, where ``g0mult`` is
            the bayesian multiplier to use on G0, and vice versa.
        """
        if p is None:
            p = self.p
        if g0prob is None:
            g0prob = self.g0prob
        if orac_val == 0:
            denom = p*g0prob+(1.0-p)*(1.0-g0prob)
            return {"G0":p/denom, "G1":(1.0-p)/denom}
        else:
            denom = (1.0-p)*g0prob+p*(1.0-g0prob)
            return {"G0":(1.0-p)/denom, "G1":p/denom}
    def ternary_oracle(self, eta=None):
        """Ternary-output oracle with parameter eta.

        :param eta: Defines input-dependent flip-or-erase
            probabilities. Defaults to ``eta``.

        :returns: If ``msgbin`` is ``0``, ``0`` with
            probability ``1-eta``, ``1`` with probability
            ``eta/2``, and an erasure otherwise. If ``msgbin``
            is ``1``, ``0`` with probability ``eta``, ``1``
            with probability ``(1-eta)/2``, and an erasure
            otherwise. Erasures represented by ``2``.
        """
        if eta is None:
            eta = self.eta
        roll = np.random.uniform()
        if self.msgbin == 0:
            if roll < 1.0-eta:
                return 0
            elif roll < 1.0-(eta/2):
                return 1
            else:
                return 2
        else:
            if roll < eta:
                return 0
            elif roll < 0.5+(eta/2):
                return 1
            else:
                return 2
    def bayesian_multipliers_ternary(self, orac_val,eta=None, g0prob=None):
        """Get multipliers for the bayesian update to be performed
        on G0 and G1 given a value from the ternary-output oracle.
        
        :param orac_val: Value given from oracle.
        :param eta: Truth value of oracle. Defaults to ``eta``.
        :g0prob: Total probability mass of G0. Defaults to ``g0prob``.

        :returns: A list ``{"G0":g0mult, "G1":g1mult}``, where ``g0mult`` is
            the bayesian multiplier to use on G0, and vice versa.
        """
        if eta is None:
            eta = self.eta
        if g0prob is None:
            g0prob = self.g0prob
        if orac_val == 0:
            denom = (1.0-eta)*g0prob+eta*(1.0-g0prob)  
            return {"G0":(1.0-eta)/denom,"G1":eta/denom}
        elif orac_val == 1:
            denom = (1.0-eta)*(1.0-g0prob)+eta*g0prob
            return {"G0":eta/denom,"G1":(1.0-eta)/denom}
        else:
            denom = (1.0-eta)*(1.0-g0prob)+eta*g0prob
            return {"G0":eta/denom,"G1":(1.0-eta)/denom}

class sorting_causal_pba(causal_setting):
    """Simulates a PBA in the causal setting that stores
    its probability mass function directly as a dictionary,
    and partitions it into bins via Dantzig's
    sorting approximation of the partition problem.
    """
    def __init__(self, p=0.9, true_message="0", q=0.9, eta=0.9):
        causal_setting.__init__(self,p,true_message,q,eta)
        self.prior = {'0':0.5, '1':0.5}
        """Dictionary representing decoder's probability belief function."""
        self.ordering = ['0','1']
        """Ordering of strings by descending probability mass."""
        self.g0cutoff=1
        """Cutoff such that ``ordering[:g0cutoff]`` is G0."""
        for msg in self.ordering[:self.g0cutoff]:
            if msg == self.accessed_msg:
                self.msgbin = 0
                break
    def sort_ordering(self):
        """Sort ordering by descending probability mass and update
        mode.
        """
        self.ordering = sorted(self.ordering, key=lambda msg: self.prior[msg],
                reverse=True)
        if self.mode != self.ordering[0]:
            self.mode = self.ordering[0] 
    def update_g0(self):
        """Assuming sorted ordering, update G0's cutoff by way of
        Dantzig's approximation to the partition problem.
        """
        cumprob = 0
        cumproblist = []
        self.msgbin = 1
        for k in range(len(self.ordering)):
            if cumprob+self.prior[self.ordering[k]] > 0.5:
                if k==0:
                    self.g0cutoff=1
                    self.g0prob = self.prior[self.ordering[0]]
                    if self.accessed_msg == self.ordering[0]:
                        self.msgbin = 0
                else:
                    self.g0cutoff=k
                    self.g0prob = math.fsum(cumproblist)
                break
            if self.msgbin ==1 and self.accessed_msg==self.ordering[k]:
                self.msgbin = 0
            cumprob+=self.prior[self.ordering[k]]
            cumproblist.append(self.prior[self.ordering[k]])
    def update_prior(self, g0mult, g1mult):
        """Perform bayesian update on prior function, given
        multipliers.
        
        :param g0mult: Multiplier to apply to G0.
        :param g1mult: Multiplier to apply to G1.
        """
        for msg in self.ordering[:self.g0cutoff]:
            self.prior[msg] = self.prior[msg]*g0mult
            if self.prior[msg] > 1.0:
                self.prior[msg] = 0.0
        for msg in self.ordering[self.g0cutoff:]:
            self.prior[msg] = self.prior[msg]*g1mult
            if self.prior[msg] > 1.0:
                self.prior[msg] = 0.0
    def next_prior(self, q=None, stepno=None):
        """Perform prior update on all messages in ``prior`` to
        generate next prior probability function.
        
        :param q: Bit arrival probability. Defaults to ``q``.
        :param stepno: Step that instance is on. Defaults to
            ``stepno``.
        """
        if q is None:
            q = self.q
        if stepno is None:
            stepno = self.stepno
        if stepno >= self.n:
            stepno = self.n
        newprior = {}
        newordering = []
        for msg in self.ordering:
            newordering.append(msg)
            if len(msg) == 1:
                newprior[msg] = (1.0-q)*self.prior[msg]
                if stepno == 1:
                    extendprob = q*0.5*self.prior[msg]
                    if extendprob > 1.0:
                        extendprob = 0.0
                    newprior[msg+"0"]=extendprob
                    newprior[msg+"1"]=extendprob
                    newordering.append(msg+"0")
                    newordering.append(msg+"1")
            elif len(msg) == stepno:
                if stepno == self.n:
                    newprior[msg] = q*0.5*self.prior[msg[:-1]]+self.prior[msg]
                else:
                    extendprob = q*0.5*self.prior[msg]
                    if extendprob > 1.0:
                        extendprob = 0.0
                    newprior[msg+"0"]=extendprob
                    newprior[msg+"1"]=extendprob
                    newprior[msg] = (1.0-q)*self.prior[msg]+q*0.5*self.prior[msg[:-1]]
                    newordering.append(msg+"0")
                    newordering.append(msg+"1")
            else:
                newprior[msg] = (1.0-q)*self.prior[msg]+q*0.5*self.prior[msg[:-1]]
            if newprior[msg] > 1.0:
                newprior[msg] = 0.0
        self.prior = newprior
        self.ordering = newordering

class tern_tree_node:
    """A node in the ternary tree representing the vocabulary
    of strings at a given step in the causal setting with
    random bit arrivals.
    """
    def __init__(self, addchar="0", parent=None,
            terminal=False,q=0.9, depthlim=1):
        self.term_prob = 1.0
        """Conditional probability that the string terminates at this
        prefix. Initially 1.
        """
        self.oneprob = 0.0
        """Conditional probability that the prefix is followed by a ``1``.
        Initially 0.
        """
        self.zeroprob = 0.0
        """Conditional probability that the prefix is followed by a ``0``.
        Initially 0.
        """
        self.onechild = None
        """``tern_tree_node`` corresponding to ``pref+"1"``.
        Initially ``None``.
        """
        self.zerochild = None
        """``tern_tree_node`` corresponding to ``pref+"0"``.
        Initially ``None``.
        """
        self.terminal = terminal
        """Indicates whether this node will never have children.
        False by default.
        """
        if parent is None:
            self.parent = None
            """Node corresponding to ``pref[:len(pref)]``. ``None``
            for root node.
            """
            self.pref = ""
            """Binary prefix represented by node. ``""`` for root node.
            """
            self.term_prob = 0.0
            self.oneprob = 0.5
            self.zeroprob = 0.5
            self.depth = 0
            """Node depth. By default equals ``len(pref)``."""
            self.last_prior_upd = 0
            """Step at which a prior update was last performed on node.
            Equals ``len(pref)`` at initialization.
            """
            self.q = q
            """Bit arrival probability for system. Defaults to parent's value
            of ``q``.
            """
            self.depthlim = depthlim
            """Depth, or prefix length limit for system.
            """
        else:
            self.parent = parent
            self.depth = parent.depth+1
            self.q = parent.q
            self.last_prior_upd = self.depth
            self.pref = parent.pref+addchar
            self.depthlim = parent.depthlim
    def update_prior(self, step, better=False, q=None,terminal=None):
        """Apply update of prior probability a number
        times equal to the number of steps elapsed since
        it was last performed.

        :param step: Number of steps that have elapsed since the
            start of the simulation.
        :param better: Use a linear-time approximation of the prior update
            if true. Defaults to false, use logarithmic-time approximation
            instead.
        :param q: Bit arrival probability. Default ``q``.

        """
        if q is None:
            q = self.q
        if terminal is None:
            terminal = self.terminal
        if (not terminal) and (self.parent is not None)\
            and (step > self.last_prior_upd):
            missed_steps = step-self.last_prior_upd
            self.last_prior_upd = step
            if better:
                if self.depth > 0:
                    newtermprob = self.term_prob
                    if self.pref[-1] == "0":
                        additive = (q*self.parent.term_prob)/(2*self.parent.zeroprob)
                    else:
                        additive = (q*self.parent.term_prob)/(2*self.parent.oneprob)
                    for iteration in range(missed_steps):
                        newtermprob = ((1.0-q)*newtermprob+additive)/(1+additive)
                        if newtermprob > 1.0:
                            newtermprob = 0.0
                            break
            else:
                newtermprob = self.term_prob*((1.0-q)**(missed_steps))
                if newtermprob > 1.0:
                    newtermprob = 0.0
            difference = (self.term_prob-newtermprob)/2
            self.oneprob += difference
            self.zeroprob += difference
            self.term_prob = newtermprob

    def getonechild(self, preterminal=False):
        """
        :returns: ``onechild`` node, creates it if it does not exist.
        """
        if self.depth == self.depthlim-1:
            preterminal = True
        if self.onechild is None:
            self.onechild = tern_tree_node("1",self,preterminal)
        return self.onechild
    def getzerochild(self, preterminal=False):
        """
        :returns: ``zerochild`` node, creates it if it does not exist.
        """
        if self.depth == self.depthlim-1:
            preterminal = True
        if self.zerochild is None:
            self.zerochild = tern_tree_node("0",self,preterminal)
        return self.zerochild 
    def normalize1(self):
        """Normalize 1-probability assuming all nodes below 1-child
        are normalized, then normalize probabilities of 1-child.
        Do nothing if 1-child DNE.
        """
        if self.onechild is not None:
            mult = self.onechild.term_prob+self.onechild.oneprob\
                +self.onechild.zeroprob
            self.oneprob*=mult
            self.onechild.term_prob /= mult
            self.onechild.oneprob /= mult
            self.onechild.zeroprob /= mult
    def normalize0(self):
        """Normalize 0-probability assuming all nodes below 0-child
        are normalized, then normalize probabilities of 0-child.
        Do nothing if 0-child DNE.
        """
        if self.zerochild is not None:
            mult = self.zerochild.term_prob+self.zerochild.oneprob\
                +self.zerochild.zeroprob
            self.zeroprob*=mult
            self.zerochild.term_prob /= mult
            self.zerochild.oneprob /= mult
            self.zerochild.zeroprob /= mult

class tree_causal_pba(causal_setting):
    """Models a PBA for the causal setting that stores its belief function
    as a ternary tree, and uses this to achieve a linear-time algorithm.
    """
    def __init__(self, p=0.9, true_message="0", q=1.0, eta=0.9, k=0,\
            better_prior=False,streaming=False):
        causal_setting.__init__(self,p,true_message,q,eta)
        self.modeprob = 0.5
        """Assigned probability to estimate."""
        self.stepno = 0
        self.tree_root = tern_tree_node(q=self.q,depthlim=self.n)
        """Root node of ternary tree representing probability mass
        function.
        """
        self.g0 = [[self.tree_root, "0"]]
        """List of edges represented in ``[node, edge_label]`` form
        of interest to G0.
        """
        self.g1 = [[self.tree_root, "1"]]
        """List of edges represented in ``[node, edge_label]`` form
        of interest to G1.
        """
        self.bottomnodes = [self.tree_root]
        """List of bottom-most nodes to renormalize after posterior
        update.
        """
        self.k=k
        """Parameter k dictating how much refinement to perform during
        the partition of G0 and G1.
        """
        self.better_prior = better_prior
        """Whether to use the more precise prior updating method. False
        by default.
        """
        self.streaming=streaming
        """Whether the model is in a streaming setting (limits estimate
        search accordingly). False by default.
        """
        if self.accessed_msg == "1":
            self.msgbin = 1
    def update_posterior(self, g0mult, g1mult):
        """Perform bayesian update on edges of interest in tree, given
        multipliers. Afterwards, renormalize tree by working from bottom-most
        nodes in ``bottomnodes`` and calling normalization functions from
        parents until the root is reached.
        
        :param g0mult: Multiplier to apply to edges in G0.
        :param g1mult: Multiplier to apply to edges in G1.
        """
        for node in self.g0:
            if node[1] == "0":
                node[0].zeroprob*=g0mult
            elif node[1] == "1":
                node[0].oneprob*=g0mult
            else:
                node[0].term_prob*=g0mult
        for node in self.g1:
            if node[1] == "0":
                node[0].zeroprob*=g1mult
            elif node[1] == "1":
                node[0].oneprob*=g1mult
            else:
                node[0].term_prob*=g1mult
        for node in self.bottomnodes:
            focusnode = node
            while focusnode.parent is not None:
                if focusnode.pref[-1] == "0":
                    focusnode.parent.normalize0()
                else:
                    focusnode.parent.normalize1()
                focusnode = focusnode.parent
    def update_msgbin(self):
        """Check whether ``accessed_msg`` is in G0 or
        G1 and update ``msgbin`` accordingly.
        """
        self.msgbin = 1
        for node in self.g0:
            if node[1] == "":
                if node[0].pref == self.accessed_msg:
                    self.msgbin = 0
            else:
                pref = node[0].pref+node[1]
                if pref == self.accessed_msg[:len(pref)]:
                    self.msgbin = 0
    def update_partition(self, step=None, searchlim=None, k=None,\
            better_prior=None,streaming=None):
        """Search tree while keeping track of the highest-probability
        string encountered as ``mode``, applying prior-update at
        each node. Add each untraveled edge to ``g1``. When either
        a search limit is reached or the probability mass of the largest-mass
        tree observed is less than or equal to 0.5, add all hitherto-encountered
        edges and corresponding subtree tree/string probability masses to a list,
        up to ``k`` edge-travels,
        sort this list, and use Dantzig's approximation to get the probability mass
        of G0 as close to 0.5 as possible.

        :param step: Step the process is on. Defaults to ``stepno``.
        :param searchlim: Search depth limit, is equal to the expected
            length of ``accessed_msg``.
        :param k: Parameter with which to refine G0. Defaults to ``k``.
        :param better_prior: Whether to use the more precise prior update.
            Defaults to ``better_prior``.
        :param streaming: Whether the model is in a streaming setting (searches
            for mode up to full possible depth if false, stops searching at
            ``searchlim`` if true.
        """
        if step is None:
            step = self.stepno
        if searchlim is None:
            searchlim = max(1,int(float(step)*self.q))
        if k is None:
            k = self.k
        if better_prior is None:
            better_prior = self.better_prior
        if streaming is None:
            streaming=self.streaming
        self.modeprob = 0.0
        self.mode = ""
        focusnode = self.tree_root
        self.g0 = []
        self.g1 = []
        treesofinterest=[]
        self.g0prob = 1.0
        self.bottomnodes = []
        ended_by_limit = True
        while focusnode.depth <= searchlim and self.g0prob > 0.5:
            focusnode.update_prior(step=step, better=better_prior)
            if self.g0prob*focusnode.term_prob > self.modeprob:
                self.mode = focusnode.pref
                self.modeprob = self.g0prob*focusnode.term_prob
            if focusnode.depth > 0:
                if focusnode.pref[-1] == "0":
                    self.g1.append([focusnode.parent, "1"])
                    self.g1.append([focusnode.parent, ""])
                else:
                    self.g1.append([focusnode.parent, "0"])
                    self.g1.append([focusnode.parent, ""])
            if (focusnode.zeroprob > focusnode.term_prob) and (focusnode.zeroprob >= focusnode.oneprob):
                self.g0prob *= focusnode.zeroprob
                focusnode = focusnode.getzerochild()
            elif (focusnode.oneprob > focusnode.term_prob) and (focusnode.oneprob > focusnode.zeroprob):
                self.g0prob *= focusnode.oneprob
                focusnode = focusnode.getonechild()
            else:
                ended_by_limit = False
                break
        focusnode.update_prior(step=step, better=better_prior)
        if self.g0prob*focusnode.term_prob > self.modeprob:
            self.mode = focusnode.pref
            self.modeprob = self.g0prob*focusnode.term_prob
        if ended_by_limit: 
            if focusnode.pref[-1] == "0":
                self.g0prob /= focusnode.parent.zeroprob
                focusnode = focusnode.parent
                treesofinterest.append([[focusnode,"1"],self.g0prob*focusnode.oneprob])
                treesofinterest.append([[focusnode,""],self.g0prob*focusnode.term_prob])
                branch_to_split = "0"
            else:
                self.g0prob /= focusnode.parent.oneprob
                focusnode = focusnode.parent
                treesofinterest.append([[focusnode,"0"],self.g0prob*focusnode.zeroprob])
                treesofinterest.append([[focusnode,""],self.g0prob*focusnode.term_prob])
                branch_to_split = "1"
        else:
            treesofinterest.append([[focusnode,""],self.g0prob*focusnode.term_prob])
            if focusnode.zeroprob >= focusnode.oneprob:
                branch_to_split = "0"
                treesofinterest.append([[focusnode, "1"],self.g0prob*focusnode.oneprob])
            else:
                branch_to_split = "1"
                treesofinterest.append([[focusnode, "0"],self.g0prob*focusnode.zeroprob])
        for iteration in range(k):
            if branch_to_split == "0":
                if focusnode.zeroprob == 0.0:
                    break
                self.g0prob *= focusnode.zeroprob
                focusnode = focusnode.getzerochild()
            else:
                if focusnode.oneprob == 0.0:
                    break
                self.g0prob *= focusnode.oneprob
                focusnode = focusnode.getonechild()
            focusnode.update_prior(step=step, better=better_prior)
            if self.g0prob*focusnode.term_prob > self.modeprob:
                self.mode = focusnode.pref
                self.modeprob = self.g0prob * focusnode.term_prob
            treesofinterest.append([[focusnode,""],self.g0prob*focusnode.term_prob])
            if focusnode.zeroprob >= focusnode.oneprob:
                branch_to_split = "0"
                treesofinterest.append([[focusnode,"1"],self.g0prob*focusnode.oneprob])
            else:
                branch_to_split = "1"
                treesofinterest.append([[focusnode,"0"],self.g0prob*focusnode.zeroprob])
        self.bottomnodes.append(focusnode)
        if branch_to_split == "0":
            treesofinterest.append([[focusnode,"0"],self.g0prob*focusnode.zeroprob])
        else:
            treesofinterest.append([[focusnode,"1"],self.g0prob*focusnode.oneprob])
        while (focusnode.depth <= step and (not streaming)) or (focusnode.depth <= searchlim):
            focusnode.update_prior(step=step,better=better_prior)
            if self.g0prob*focusnode.term_prob > self.modeprob:
                self.mode = focusnode.pref
                self.modeprob = self.g0prob*focusnode.term_prob
            if (focusnode.zeroprob > focusnode.term_prob) and (focusnode.zeroprob >= focusnode.oneprob):
                self.g0prob *= focusnode.zeroprob
                focusnode = focusnode.getzerochild()
            elif (focusnode.oneprob > focusnode.term_prob) and (focusnode.oneprob > focusnode.zeroprob):
                self.g0prob *= focusnode.oneprob
                focusnode = focusnode.getonechild()
            else:
                break
        treesofinterest = sorted(treesofinterest, key=lambda treetuple: treetuple[1],
                reverse=True)
        self.g0prob = 0.0
        g0probvals = [0.0]
        while len(treesofinterest) > 0:
            if abs(0.5-(self.g0prob+treesofinterest[0][1])) < abs(0.5-self.g0prob):
                g0probvals.append(treesofinterest[0][1])
                self.g0prob = math.fsum(g0probvals)
                self.g0.append(treesofinterest[0][0])
            else:
                self.g1.append(treesofinterest[0][0])
            treesofinterest.pop(0)

def bsc_capacity(p):
    """The capacity of a BSC with crossover probability ``p``.

    :param p: BSC crossover probability.

    :returns: ``1-H(p)``, where ``H(p)`` is the Shannon entropy
        function.
    """
    return 1.0+p*np.log2(p)+(1.0-p)*np.log2(1.0-p)

def prob_max_msglen(i,q,n):
    """Probability that the message in a given stochastic causal
    setting has reached its maximum length. Uses a Poisson
    distribution with mean ``i*q`` for message lengths.
    
    :param i: Step number.
    :param q: Bit arrival probability.
    :param n: Maximum message length.

    :returns: The probability that the maximum message length
        has been reached.
    """
    lam = q*i
    const_val = np.exp(-lam)
    sum_vals = [0.0]
    fact = 1
    summand = 1.0
    for length in range(n):
        sum_vals.append(summand)
        summand*=(lam/fact)
        fact+=1
    return max(1.0-(const_val*math.fsum(sum_vals)),0.0)

def streaming_cap(i,p,q):
    """Effective average capacity of a BSC in the causal setting
    when the maximum message length has not been reached. Models
    the length of ``accessed_msg`` (an upper bound on the number
    of transmittable bits of information) using a Poisson distribution
    with mean ``i*q`` cut off at ``(1-H(p))i`` (channel capacity).

    :param i: Step number.
    :param p: BSC crossover probability.
    :param q: Bit arrival probability.

    :returns: The effective average capacity.
    """
    max_cap = bsc_capacity(p)
    maxbits = math.floor(i*max_cap)
    sum_probs = [0.0]
    weighted_rates = [1.0]
    lam = q*i
    fact = 1
    summand_prob = 1.0
    const_val = np.exp(-lam)
    for length in range(maxbits+1):
        sum_probs.append(summand_prob)
        weighted_rates.append(summand_prob*float(length))
        summand_prob*=(lam/fact)
        fact+=1
    maxprob = 1.0-(const_val*math.fsum(sum_probs))
    return const_val*math.fsum(weighted_rates)/i+maxprob*max_cap

def avg_cap(i,p,q,n):
    """Effective average capacity at a given step in a given
    causal setting.
    
    :param i: Step number.
    :param p: BSC crossover probability.
    :param q: Bit arrival probability.
    :param n: Maximum message length.

    :returns: The effective average capacity.
    """
    max_cap = bsc_capacity(p)
    max_prob = prob_max_msglen(i,q,n)
    nonmax_cap = streaming_cap(i,p,q)
    return max_prob*max_cap+(1.0-max_prob)*nonmax_cap

def avg_rate(p,q,n):
    """Average upper bound on rate in a given
    length-limited causal setting.
    
    :param p: BSC crossover probability.
    :param q: Bit arrival probability.
    :param n: Maximum message length.

    :returns: The average upper bound on rate.
    """
    stepnum = 0
    summed_info = [0.0]
    while True:
        stepnum+=1
        summed_info.append(avg_cap(stepnum,p,q,n))
        if math.fsum(summed_info) >= float(n):
            return float(n)/float(stepnum)

def randombinstring(length):
    """Generates a random binary string of a given length.

    :param length: Length of generated string.

    :returns: A radnom binary string of length ``length``.
    """
    string = ""
    for i in range(length):
        bit = np.random.binomial(size=None,n=1,p=0.5)
        string+=str(bit)
    return string

def hammstring(binstring1, binstring2, normalized=False, percentcheck=1.0):
    """Measures the hamming distance between two strings in various metrics.
    
    :param binstring1: A string.
    :param binstring2: Another string.
    :param normalized: If false (default) output absolute hamming distance.
        If true, output hamming distance normalized on number of characters
        checked.
    :param percentcheck: What ratio of the characters of the two strings
        (starting from the beginning) of each string to check.

    :returns: Either the absolute or normalized hamming distance between the
        leading characters of interest of the two strings.
    """
    max_index = int(percentcheck*max(len(binstring1),len(bingstring2)))
    checkstring1 = binstring1[:max_index]
    checkstring2 = binstring2[:max_index]
    hamm = abs(len(checkstring1)-len(checkstring2))
    for j in range(min(len(checkstring1),len(checkstring2))):
        if checkstring1[j] != checkstring2[j]:
            hamm+=1
    if normalized:
        return float(hamm)/float(max(len(checkstring1),len(checkstring2)))
    else:
        return hamm
