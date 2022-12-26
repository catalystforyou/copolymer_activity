import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.colors
import matplotlib.ticker

def visualize3D():
    data = pd.read_csv('data/trainval.csv')
    x = data['A1']
    y = data['A2']
    z = data['A3']
    f = data['Activity']
    min_f = min(f)
    max_f = max(f)
    color = [plt.get_cmap("rainbow", 100)(int(float(i-min_f)/(max_f-min_f)*100)) for i in f]
    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection ="3d")
    plt.set_cmap(plt.get_cmap("rainbow", 100))
    im = ax.scatter(x, y, z, s=100,c=color,marker='.')
    fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_f-min_f)+min_f)))
    ax.set_xlabel('A1')
    ax.set_ylabel('A2')
    ax.set_zlabel('A3')
    plt.savefig("data.png")

def visualize2D():
    data = pd.read_csv('data/trainval.csv')
    pool = ['A1','A2','A3','A4']
    f = data['Activity']
    min_f = min(f)
    max_f = max(f)
    color = [plt.get_cmap("rainbow", 100)(int(float(i-min_f)/(max_f-min_f)*100)) for i in f]
    for ax1 in range(4):
        for ax2 in range(ax1+1,4):
            x = data[pool[ax1]]
            y = data[pool[ax2]]
            fig = plt.figure(figsize=(15,10))
            ax = plt.axes()
            plt.set_cmap(plt.get_cmap("rainbow", 100))
            im = ax.scatter(x, y, s=100,c=color,marker='.')
            fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_f-min_f)+min_f)))
            ax.set_xlabel(pool[ax1])
            ax.set_ylabel(pool[ax2])
            figname = pool[ax1] + pool[ax2] + '.png'
            plt.savefig(figname)

if __name__ == '__main__':
    visualize3D()
    visualize2D()
