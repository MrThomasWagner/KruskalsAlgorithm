
## Implementing Kruskal's Algorithm

import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import igraph
import mpl_toolkits.mplot3d


#We are going to randomnly generate the vertices, so we need a function to get the distances between each and all of them
def generate_edges_from_vertices(vertices):
    length = len(vertices)
    edges = list()
    
    for k,v in vertices.iteritems():        
        for k2,v2 in {n:m for n,m in vertices.iteritems() if n > k }.iteritems():
            edges.append(((k, k2), get_dist(vertices.get(k), vertices.get(k2))))
            
    return edges


#arbitrary dimension euclidean distance
def get_dist(a, b):
    return math.sqrt(sum([(a[x]-b[x])**2 for x in range(len(a))]))


#Kruskal's algorithm
def kruskal(vertices, edges):
    final_edges = []
    sorted_edges = sorted(edges, key=operator.itemgetter(1))
    current_trees = [set([k]) for k in vertices.iterkeys()]
    
    while len(current_trees) > 1:
        next_edge = sorted_edges.pop(0)
        current_vertices = next_edge[0]
        
        p1, p2 = current_vertices[0],current_vertices[1]
        
        valid = True
        for tree in current_trees:
            if (p1 in tree) and (p2 in tree):
                valid = False
                break
                
        if valid:
            set_a = None
            set_b = None
        
            final_edges.append(next_edge[0])
            
            for tree in current_trees:
                if (p1 in tree):
                    set_a = tree
                if (p2 in tree):
                    set_b = tree
        
            new_set = set_a.union(set_b)

            current_trees.remove(set_a)
            current_trees.remove(set_b)
            current_trees.append(new_set)
            
    return final_edges


#helper func for plotting edge lists
def plot_edges(edge_list, vertices, dimensions, ax):
    for origin,dest in edge_list:
        ax.plot(*[[vertices.get(origin)[x], vertices.get(dest)[x]] for x in range(dimensions)])


#Go! 2 dimensions
dimensions = 2
arr = np.random.random((30, dimensions))
vertices = dict(enumerate(arr))
edges=generate_edges_from_vertices(vertices)

#Plotting the two dimension result
edge_list = kruskal(vertices, edges)

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.scatter(arr[:,0], arr[:,1])

ax2 = fig.add_subplot(122)
plot_edges(edge_list, vertices, dimensions, ax2)

ax2.scatter(arr[:,0], arr[:,1])

fig.show()


## Lets try with three dimensions...

dimensions = 3
num_vertices = 30
arr = np.random.random((num_vertices, dimensions))
vertices = dict(enumerate(arr))
edges=generate_edges_from_vertices(vertices)


fig = plt.figure(figsize=(24,8))

#scatter, no edges
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(arr[:,0], arr[:,1], arr[:,2])

#get the min-span-tree edges
edge_list = kruskal(vertices, edges)

ax2 = fig.add_subplot(122, projection='3d')
plot_edges(edge_list, vertices, dimensions, ax2)

ax2.scatter(arr[:,0], arr[:,1], arr[:,2])

fig.show()
## Interactive plot from plotly

# In[8]:

import plotly.plotly as py
from plotly.graph_objs import *

py.sign_in('your username', 'your api key')

dimensions = 3
arr = np.random.random((30, dimensions))
vertices = dict(enumerate(arr))
edges=generate_edges_from_vertices(vertices)

edge_list = kruskal(vertices, edges)
data = []
for origin, dest in edge_list:
    coors = [[vertices.get(origin)[x], vertices.get(dest)[x]] for x in range(dimensions)]
    
    trace = Scatter3d(
        x=coors[0],
        y=coors[1],
        z=coors[2]
    )
    
    data.append(trace)

#py.plot(data)


# In[ ]:



