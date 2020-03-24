import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib.colors as colors

def euclidean(x, y):
    return math.sqrt((x['x']-y['x'])**2 + (x['y']-y['y'])**2)

def normalization(data):
    nmin = {'x': min([n['x'] for n in data]), 'y': min([n['y'] for n in data])}
    nmax = {'x': max([n['x'] for n in data]), 'y': max([n['y'] for n in data])}
    for i in data:
        i['x'] = (i['x'] - nmin['x']) / (nmax['x'] - nmin['x'])
        i['y'] = (i['y'] - nmin['y']) / (nmax['y'] - nmin['y'])
    return data

def plot(koor,z):
    koor.set(xlabel='x', ylabel='y', title=z)
    koor.grid()
    plt.axis([0,1,0,1])
    plt.show()

def dataLuar():
    reader = csv.reader(open("Dataset.csv"), delimiter=",")
    x = list(reader)
    return np.array(x)

data = []
datas = dataLuar()
for i in datas:
    data.append({'x': float(i[0]), 'y': float(i[1]), 'color': "red"})
clusters = []

color = []
for name, hex in colors.cnames.items():
    color.append(name)

lr = 0.000000000000000001
tLr = 2

sig = 2
tsig = 2

data_normal = normalization(data)

neuron = []
neuron_size = 1400
for i in range(neuron_size):
    neuron.append({'x': random.uniform(0, 1), 'y': random.uniform(0, 1), 'color': 'limegreen', 'status': ''})

fig, ax = plt.subplots()
for i in neuron:
    ax.plot(i['x'], i['y'], ".", color=i['color'])
for i in data_normal:
    ax.plot(i['x'], i['y'], ".", color=i['color'])

plot(ax,'Data Awal')

iterasi = 50
konvergen = 0.0000000000000000000001
win_neuron = neuron[0]
for t in range(iterasi):
    rand = random.randint(1, len(data)-1)
    x = data[rand]

    for first in neuron:
        first['status'] = 'neuron'
        if euclidean (x, first) < euclidean (x, win_neuron): win_neuron = first

    for second in neuron:
        if euclidean(second, win_neuron) < sig:
            second['status'] = 'neighborhood'
            s = euclidean(win_neuron, second)
            phi = np.exp(-(s**2 / (2*sig**2)))
            weight = lr * phi * euclidean(x, second)
            second['x'] += weight
            second['y'] += weight

    if (weight < konvergen):
        print("iterasi: ",t)
        break

    if (win_neuron not in clusters):
        clusters.append(win_neuron)
        win_neuron['color'] = color[t+10]
        lr *= np.exp(-t/tLr)
        tsig *= np.exp(-t/tsig)

win_clusters = clusters[0]
for d in data_normal:
    for cluster in clusters:
        if euclidean(d, cluster) < euclidean(d, win_clusters): win_clusters = cluster
    d['color'] = win_clusters['color']

print("Total clusters: ", len(clusters))

figure, axx = plt.subplots()
for i in data_normal:
    axx.plot(i['x'], i['y'], ".", color=i['color'])
plot(axx,'Data Akhir')
