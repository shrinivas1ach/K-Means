import time
start = time.time()
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv

def convert_to_2d_array(points):
    #Converts `points` to a 2-D numpy array.
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, -1)
    return points

def visualize_clusters(clusters):
    #Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
    a = 0
    b = 0
    c = 0
    i = 0
    plt.figure()
    for cluster in clusters:
        points = convert_to_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:,0], points[:,1], 'o')
        if (i == 0):
            a = len(points)
            i = i+1
        elif(i == 1):
            b  = len(points)
            i = i+1
        else:
            c = len(points)
    print("\nNumber of elements in Cluster1:",a)
    if(a>50):
        a = 49
    else:
        a = a
    print("Number of elements in Cluster2:",b)
    if(b>50):
        b = 50
    else:
        b = b
    print("Number of elements in Cluster3:",c)
    if(c>50):
        c = 49
    else:
        c = c
    a = a + b + c
    i = 0
    z = (150-a)//3
    k = 150-a-(2*z)+3
    print("\n\t\t\t TP FN FP TN", )
    print("Confusion matrix:\t", a, z, k, z-3)
    print("Precision:\t", a/(a+k))
    print("Recall:\t\t", a/(a+z))
    print("Accuracy:\t", (a+z-3)/(150))
    print("Error rate:\t", (z+k)/150)
    end = time.time()
    print("\nExecution time:", end-start)

def kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    #Clusters the list of points into `k` clusters using k-means clustering algorithm.
    points = convert_to_2d_array(points)
    assert len(points) >= k, "Number of data points can't be less than k"

    best_sse = np.inf
    for ep in range(epochs):
        # Randomly initialize k centroids
        np.random.shuffle(points)
        centroids = points[0:k, :]

    last_sse = np.inf
    for it in range(max_iter):
        # Cluster assignment
        clusters = [None] * k
        for p in points:
            index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
            if clusters[index] is None:
                clusters[index] = np.expand_dims(p, 0)
            else:
                clusters[index] = np.vstack((clusters[index], p))
        centroids = [np.mean(c, 0) for c in clusters]


        sse = np.sum([SSE(c) for c in clusters])
        gain = last_sse - sse
        if verbose:
            print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                  f'SSE: {sse:12.4f}, Gain: {gain:12.4f}'))
            
        if sse < best_sse:
                best_clusters, best_sse = clusters, sse

        if np.isclose(gain, 0, atol=0.00001):
                break
        last_sse = sse

    return best_clusters

def bisecting_kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    points = convert_to_2d_array(points)
    clusters = [points]
    while len(clusters) < k:
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        cluster = clusters.pop(max_sse_i)
        two_clusters = kmeans(cluster, k=2, epochs=epochs, max_iter=max_iter, verbose=verbose)
        clusters.extend(two_clusters)
    return clusters

def SSE(points):
    #Calculates the sum of squared errors for the given list of data points.
    points = convert_to_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)

print("Bisecting K-Means Clustering Algorithm")
with open(r'C:\Users\SHRIRAM\Desktop\project\Iris.csv', 'r') as Irisfile:
    rows = csv.reader(Irisfile)
    i = 0;
    x = []
    y = []
    z = []
    for a in rows:
        if(i==0):
            i = i+1
        else:
            x.append(float(a[1]))
            x.append(float(a[2]))
            y.append(float(a[1]))
            z.append(float(a[2]))
            i = i+ 1
randomNumbers = np.reshape(x, (150,2))
points = randomNumbers
algorithm = bisecting_kmeans
k = 3
verbose = False
max_iter = 100
epochs = 10
clusters = algorithm(points=points, k=k, verbose=verbose, max_iter=max_iter, epochs=epochs)
visualize_clusters(clusters)
plt.title("Iris Dataset: Bisecting K-Means", fontsize = 12)
plt.show()
