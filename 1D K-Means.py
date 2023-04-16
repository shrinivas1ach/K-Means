import matplotlib.pyplot as plt
import numpy as np
import random
def fun(randomlist, m1, m2, iteration): #assigning clusters as per mean 
    m1 = rrr[0]
    m2 = rrr[1]
    C1 = []
    C2 = []
    iteration = iteration +1
    for i in range(0, len(randomlist)):
        a = abs(randomlist[i]-rrr[0])
        b = abs(randomlist[i]-rrr[1])
        if(a>=b):
            y = randomlist[i]
            C2.append(y)
        else:
            x = randomlist[i]
            C1.append(x)
    fun1(m1, C1, C2, iteration)
    
def fun1(m1, C1, C2, iteration): #computing new mean and clusters
    su = 0
    Cluster=C1
    print("ITERATION:", iteration)
    for i in range(0, len(C1)):
        su = su + C1[i]
        rrr[0] = su/len(C1)
    print("C1 -->", C1)
    print("M1' = ", rrr[0])
    su = 0
    for i in range(0, len(C2)):
        su = su + C2[i]
        rrr[1] = su/len(C2)
    print("C2 -->", C2)
    print("M2' = ", rrr[1])
    if(m1!=rrr[0]): #terminating condition if means are equal
        print("\n")
        fun(randomlist, m1, m2, iteration)
    elif(Cluster!=C1): #terminating condition if clusters are equal
        print("\n")
        fun(randomlist, m1, m2, iteration)
    elif(iteration==12): #stopping condition, if means and clusters aren't equal
        print("\n")
        fun(randomlist, m1, m2, iteration)
        
    
randomlist = random.sample(range(0, 80), 60) #generates random list of array
print(randomlist)
rrr = random.choices(randomlist, k=2) #selects two random points
m1= rrr[0]
m2 = rrr[1]
if(m1==m2):
    print('Compile time error')
print("Mean1: M1= ", rrr[0])
print("Mean2: M2= ", rrr[1])
iteration = 0
fun(randomlist, m1, m2, iteration)
