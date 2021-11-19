# Author: Archit Thakur
import csv
import math
import scipy.cluster.hierarchy as sp
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import sys

"""
load_data:
organizes pokemon stats into a list of dictionaries
that will be interpreted in the methods below.
"""
def load_data(filepath):
    data = [{} for ran in range(20)]
    with open(filepath, 'r', encoding='utf-8') as csvFile:
        reader = csv.DictReader(csvFile)
        i = 0
        for rows in reader:
            if (i == 20):
                break
            data[i]['#'] = int(rows['#'])
            data[i]['Name'] = rows['Name']
            data[i]['Type 1'] = rows['Type 1']
            data[i]['Type 2'] = rows['Type 2']
            data[i]['Total'] = int(rows['Total'])
            data[i]['HP'] = int(rows['HP'])
            data[i]['Attack'] = int(rows['Attack'])
            data[i]['Defense'] = int(rows['Defense'])
            data[i]['Sp. Atk'] = int(rows['Sp. Atk'])
            data[i]['Sp. Def'] = int(rows['Sp. Def'])
            data[i]['Speed'] = int(rows['Speed'])
            i = i + 1
    return data

"""
calculate_x_y:
a formula used to sum the relevant stats that represent
the pokemon's total attack and defense, returned as a tuple.
"""
def calculate_x_y(stats):
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return (x, y)

"""
hac:
the main method for performing heirarchical agglomerative 
clustering; the tree builds from the bottom up, and the
matrix result is constructed in a way such that the
information can be easily referred to recursively, specifically
in imshow_helper below.
The algorithm is as follows:
   -If the data point is a leaf (data points very close to
    each other), assign an index to the leaf and its predecessor.
   -If a data point is not a leaf, its index is represented by
    the index of its child +dataset range, represented by m
    (allows for recursive interpretation later)
   -A tie case regarding equidistant points arbitrarily chooses
    one connection over the other; this could be improved on further.
"""
def hac(dataset):
    dataset = [i for i in dataset if not math.isnan(i[0]) and math.isfinite(i[0])
        and not math.isnan(i[1]) and math.isfinite(i[1])]
    m = len(dataset)
    point_set = {}
    for i in range(m):
        point_set[str(i)] = None

    #initialises the final cluster set
    cluster = [[None for x in range(4)] for y in range(m-1)]
    #Tracka the pairs which are already used
    was_used = [[None for x in range(3)] for y in range(m-1)]

    for row in range(m - 1):
        max_pts = [math.inf, math.inf, math.inf]
        #The shortest distance is found
        for i in range(m):
            for j in range(m):
                if ((point_set[str(i)] != point_set[str(j)]) or (point_set[str(i)] == None and point_set[str(j)] == None)):
                    dup = False
                    d = math.dist(dataset[i], dataset[j])
                    for k in range(row):
                        if ((i == was_used[k][1] and j == was_used[k][2]) or
                            (i == was_used[k][2] and j == was_used[k][1])):
                            dup = True
                    if (i != j and d < max_pts[0] and not dup):
                        max_pts[0] = d
                        max_pts[1] = i
                        max_pts[2] = j
                    elif (i != j and d == max_pts[0] and not dup):
                        #tie case
                        if ((i < max_pts[1] and i < max_pts[2]) or ((j < max_pts[1] and j < max_pts[2]))):
                            max_pts[0] = d
                            max_pts[1] = i
                            max_pts[2] = j
                        elif ((i < max_pts[1] and i == max_pts[2] and j < max_pts[1]) or
                            (i < max_pts[2] and i == max_pts[1] and j < max_pts[2]) or
                            (j < max_pts[1] and j == max_pts[2] and i < max_pts[1]) or
                            (j < max_pts[2] and j == max_pts[1] and i < max_pts[2])):
                            max_pts[0] = d
                            max_pts[1] = i
                            max_pts[2] = j
        was_used[row][0] = max_pts[0]
        was_used[row][1] = max_pts[2]
        was_used[row][2] = max_pts[1]
        c1 = point_set[str(max_pts[1])]
        c2 = point_set[str(max_pts[2])]
        #new clusters for point
        p1 = 0
        p2 = 0
        #tot includes all pts in clusters
        tot = 2
        if (c1 == None):
            p1 = max_pts[1]
            point_set[str(max_pts[1])] = row
        else:
            p1 = c1 + m
            tot = tot + cluster[c1][3] - 1
            for num in point_set:
                if (point_set[num] == c1):
                    point_set[num] = row
        if (c2 == None):
            p2 = max_pts[2]
            point_set[str(max_pts[2])] = row
        else:
            p2 = c2 + m
            tot = tot + cluster[c2][3] - 1
            for num in point_set:
                if (point_set[num] == c2):
                    point_set[num] = row
        if (p1 < p2):
            cluster[row][0] = p1
            cluster[row][1] = p2
        else:
            cluster[row][1] = p1
            cluster[row][0] = p2
        cluster[row][2] = max_pts[0]
        cluster[row][3] = tot

    return np.matrix(cluster)

"""
random_x_y:
a debugging method to test the breadth of the program.
Returns a list identical in structure to the one returned by
load_data.
"""
def random_x_y(m):
    rand_list = [None for x in range(m)]
    for i in range(m):
        rand_list[i] = (rnd.randint(1, 359), rnd.randint(1, 359))
    return rand_list
"""
imshow_hac:
plots the points in dataset by cascading
up, plotting the lower points first then joining each
point with a line to make the data more readable.
"""
def imshow_hac(dataset):
    cluster_set = hac(dataset)
    m = len(cluster_set) + 1
    fig, ax= plt.plot(1, 2, figsize=(10,4))
    scatter_set = np.transpose(np.array(dataset))
    empty_list = [None for x in range(m)]
    color_list = [('#'+''.join([rnd.choice('0123456789ABCDEF') for x in range(6)])) for j in empty_list]
    ax[0].scatter(scatter_set[0], scatter_set[1], color= color_list)
    #ax[1].scatter(scatter_set[0], scatter_set[1], color= color_list)

    for row in range(m - 1):
        indices = [set(), set()] #list of sets
        n1 = int(cluster_set[row, 0])
        n2 = int(cluster_set[row, 1])
        if (n1 < m):
            indices[0].add(n1)
        if (n2 < m):
            indices[1].add(n2)
        if (n1 >= m):
            indices[0] = imshow_helper(cluster_set, n1 - m, m, indices[0])
        if (n2 >= m):
            indices[1] = imshow_helper(cluster_set, n2 - m, m, indices[1])
        dist_found = False #can exit double loop when found
        for i in indices[0]: #left index
            for j in indices[1]: #right index
                ni = int(i)
                nj = int(j)
                iarr = np.array([math.dist(dataset[ni], dataset[nj])])
                jarr = np.array([cluster_set[row, 2]])
                if (np.isclose(iarr, jarr)):
                    dist_found = True
                    plt_mat = np.transpose(np.array([dataset[ni], dataset[nj]]))
                    ax[0].plot(plt_mat[0], plt_mat[1])
                    plt.pause(0.1)
                    break
            if (dist_found):
                break
    fig.delaxes(ax[1])
    plt.show()
    return None


"""
imshow_helper:
terminating recursive method
after the leaf in a tree is found.
"""
def imshow_helper(cluster_set, row, m, indices):
    n1 = int(cluster_set[row, 0])
    n2 = int(cluster_set[row, 1])
    if (n1 < m):
        indices.add(n1)
    else:
        indices = imshow_helper(cluster_set, n1 - m, m, indices)
    if (n2 < m):
        indices.add(n2)
    else:
        indices = imshow_helper(cluster_set, n2 - m, m, indices)
    return indices

"""
Driver for handling custom user input based around
the base and offset in Pokemon.csv. Allows the user to
test every range in the dataset and observe the
relations involved through imshow.
"""
if __name__=="__main__": #example simulation
    if len(sys.argv) < 3:
        print("Usage: python3 ./pokemon_stats.py <base> <offset>")
        exit()
    numPokemon = int(sys.argv[1])
    offset = int(sys.argv[2])
    if numPokemon + offset > 721:
        print("Error: base + offset is too big!")
        exit()
    dataset = load_data("Pokemon.csv")
    stats = [i for i in range(offset)]
    for j in range(offset):
        stats[j] = calculate_x_y(dataset[j+numPokemon])
    imshow_hac(stats)

