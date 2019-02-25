import monkdata as m
import dtree as d
import drawtree_qt5 as q
import random
import numpy as np
import matplotlib.pyplot as plt

# assignment 0
# dataset 2 since it's hard to chose a good split

entmonk1 = d.entropy(m.monk1)
entmonk2 = d.entropy(m.monk2)
entmonk3 = d.entropy(m.monk3)

print(str(entmonk1) + " " + str(entmonk2) + " " + str(entmonk3))

# assignment 1
# monk1 = 1.0
# monk2 = 0.957117428264771
# monk3 = 0.9998061328047111

#assignment 2
# Explain entropy for a uniform distribution and a non-uniform distribution, present some example distributions
# with high and low entropy.
#
# uniform = maximal entropy like a fair dice or a fair coin toss
# non uniform = lower entropy like an unfair dice or

a1gain = d.averageGain(m.monk1, m.attributes[0])
a2gain = d.averageGain(m.monk1, m.attributes[1])
a3gain = d.averageGain(m.monk1, m.attributes[2])
a4gain = d.averageGain(m.monk1, m.attributes[3])
a5gain = d.averageGain(m.monk1, m.attributes[4])
a6gain = d.averageGain(m.monk1, m.attributes[5])

print("monk1 a1: " + str(a1gain))
print("monk1 a2: " + str(a2gain))
print("monk1 a3: " + str(a3gain))
print("monk1 a4: " + str(a4gain))
print("monk1 a5: " + str(a5gain))
print("monk1 a6: " + str(a6gain) + "\n")

a1gain = d.averageGain(m.monk2, m.attributes[0])
a2gain = d.averageGain(m.monk2, m.attributes[1])
a3gain = d.averageGain(m.monk2, m.attributes[2])
a4gain = d.averageGain(m.monk2, m.attributes[3])
a5gain = d.averageGain(m.monk2, m.attributes[4])
a6gain = d.averageGain(m.monk2, m.attributes[5])

print("monk2 a1: " + str(a1gain))
print("monk2 a2: " + str(a2gain))
print("monk2 a3: " + str(a3gain))
print("monk2 a4: " + str(a4gain))
print("monk2 a5: " + str(a5gain))
print("monk2 a6: " + str(a6gain) + "\n")

a1gain = d.averageGain(m.monk3, m.attributes[0])
a2gain = d.averageGain(m.monk3, m.attributes[1])
a3gain = d.averageGain(m.monk3, m.attributes[2])
a4gain = d.averageGain(m.monk3, m.attributes[3])
a5gain = d.averageGain(m.monk3, m.attributes[4])
a6gain = d.averageGain(m.monk3, m.attributes[5])

print("monk3 a1: " + str(a1gain))
print("monk3 a2: " + str(a2gain))
print("monk3 a3: " + str(a3gain))
print("monk3 a4: " + str(a4gain))
print("monk3 a5: " + str(a5gain))
print("monk3 a6: " + str(a6gain) + "\n")

#assignment 3

# information gain for monk1
#monk1 a1: 0.07527255560831925
#monk1 a2: 0.005838429962909286
#monk1 a3: 0.00470756661729721
#monk1 a4: 0.02631169650768228
#monk1 a5: 0.28703074971578435
#monk1 a6: 0.0007578557158638421

#monk2 a1: 0.0037561773775118823
#monk2 a2: 0.0024584986660830532
#monk2 a3: 0.0010561477158920196
#monk2 a4: 0.015664247292643818
#monk2 a5: 0.01727717693791797
#monk2 a6: 0.006247622236881467
#
#monk3 a1: 0.007120868396071844
#monk3 a2: 0.29373617350838865
#monk3 a3: 0.0008311140445336207
#monk3 a4: 0.002891817288654397
#monk3 a5: 0.25591172461972755
#monk3 a6: 0.007077026074097326

#assignment 4

# the entropy of the subsets is minimized when the information gain is maximized
# minimized entropy for a subset is optimal since low entropy means it's leaning towards a decision

monk1a5_1 = d.select(m.monk1, m.attributes[4], 1)
monk1a5_2 = d.select(m.monk1, m.attributes[4], 2)
monk1a5_3 = d.select(m.monk1, m.attributes[4], 3)
monk1a5_4 = d.select(m.monk1, m.attributes[4], 4)

a1gain = d.averageGain(monk1a5_1, m.attributes[0])
a2gain = d.averageGain(monk1a5_1, m.attributes[1])
a3gain = d.averageGain(monk1a5_1, m.attributes[2])
a4gain = d.averageGain(monk1a5_1, m.attributes[3])
a6gain = d.averageGain(monk1a5_1, m.attributes[5])

print("monk1_atr5_val1_gain a1: " + str(a1gain))
print("monk1_atr5_val1_gain a2: " + str(a2gain))
print("monk1_atr5_val1_gain a3: " + str(a3gain))
print("monk1_atr5_val1_gain a4: " + str(a4gain))
print("monk1_atr5_val1_gain a6: " + str(a6gain) + "\n")

a1gain = d.averageGain(monk1a5_2, m.attributes[0])
a2gain = d.averageGain(monk1a5_2, m.attributes[1])
a3gain = d.averageGain(monk1a5_2, m.attributes[2])
a4gain = d.averageGain(monk1a5_2, m.attributes[3])
a6gain = d.averageGain(monk1a5_2, m.attributes[5])

print("monk1_atr5_val2_gain a1: " + str(a1gain))
print("monk1_atr5_val2_gain a2: " + str(a2gain))
print("monk1_atr5_val2_gain a3: " + str(a3gain))
print("monk1_atr5_val2_gain a4: " + str(a4gain))
print("monk1_atr5_val2_gain a6: " + str(a6gain) + "\n")

a1gain = d.averageGain(monk1a5_3, m.attributes[0])
a2gain = d.averageGain(monk1a5_3, m.attributes[1])
a3gain = d.averageGain(monk1a5_3, m.attributes[2])
a4gain = d.averageGain(monk1a5_3, m.attributes[3])
a6gain = d.averageGain(monk1a5_3, m.attributes[5])

print("monk1_atr5_val3_gain a1: " + str(a1gain))
print("monk1_atr5_val3_gain a2: " + str(a2gain))
print("monk1_atr5_val3_gain a3: " + str(a3gain))
print("monk1_atr5_val3_gain a4: " + str(a4gain))
print("monk1_atr5_val3_gain a6: " + str(a6gain) + "\n")

a1gain = d.averageGain(monk1a5_4, m.attributes[0])
a2gain = d.averageGain(monk1a5_4, m.attributes[1])
a3gain = d.averageGain(monk1a5_4, m.attributes[2])
a4gain = d.averageGain(monk1a5_4, m.attributes[3])
a6gain = d.averageGain(monk1a5_4, m.attributes[5])

print("monk1_atr5_val4_gain a1: " + str(a1gain))
print("monk1_atr5_val4_gain a2: " + str(a2gain))
print("monk1_atr5_val4_gain a3: " + str(a3gain))
print("monk1_atr5_val4_gain a4: " + str(a4gain))
print("monk1_atr5_val4_gain a6: " + str(a6gain) + "\n")


#q.drawTree(d.buildTree(m.monk1, m.attributes, 2))
print("Etrain")
t=d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1))

t=d.buildTree(m.monk2, m.attributes)
print(d.check(t, m.monk2))

t=d.buildTree(m.monk3, m.attributes)
print(d.check(t, m.monk3))

#assignment 5

#monk1 test entropy = 0.8287037037037037
#monk2 test entropy = 0.6921296296296297
#monk3 test entropy = 0.9444444444444444

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

#monk1train, monk1val = partition(m.monk1, 0.6)

#monk1traintree = d.buildTree(monk1train, m.attributes)

#assignment 6 simpler model = higher b ias and lower variance. pruning is used to lower variance
def prune(traintree, valtree):
    trees = []
    chosentree = []
    pruned = d.allPruned(traintree)
    status = d.check(traintree, valtree)
    for i in range(len(pruned)):
        trees.append(pruned[i])
        chosentree.append(d.check(pruned[i], valtree))
        #print(d.check(pruned[i], valtree))
        print(chosentree)
    if status <= max(chosentree):
        #print(str(status) + " is less than " + str(max(chosentree)))
        #print("now pruning tree nr: " + str(chosentree.index(max(chosentree))))
        return prune(trees[chosentree.index(max(chosentree))], valtree)
    else:
        return traintree

#print(prune(monk1traintree))
plotresult = []
std = []
def getmean(tree, val):
    i = 0
    sum = 0
    k = 3
    res = 0
    reslist = []

    while k < 9:
        while i < 5000:
            treetrain, treeval = partition(tree, k/10)
            thetree = d.buildTree(treetrain, m.attributes)
            train = prune(thetree, treeval)
            res = d.check(train, val)
            reslist.append(res)
            sum += res
            i += 1
        plotresult.append(1-(sum/5000))
        print("mean for k = " + str(k/10) + " = " + str(1-(sum/5000)) + " and standard deviation of " + str(np.std(reslist)))
        k += 1
        i = 0
        res = 0
        sum = 0
        std.append(np.std(reslist))

getmean(m.monk1, m.monk1test)
print(plotresult)
fig, ax = plt.subplots()

ax.errorbar([0.3,0.4,0.5,0.6,0.7,0.8], plotresult, std, fmt="ro-", capsize=5)
plotresult = []
std = []
getmean(m.monk3, m.monk3test)
ax.errorbar([0.3,0.4,0.5,0.6,0.7,0.8], plotresult, std, fmt="bo-", capsize=5)
#plt.plot([0.3,0.4,0.5,0.6,0.7,0.8], plotresult, 'ro-')
plt.axis([0, 1.0, 0, 0.5])
ax.legend(["MONK-1", "MONK-3"])
plt.grid(True)
plt.xlabel("Fraction")
plt.ylabel("Mean error rate (n = 5000, stdev)")
plt.show()

