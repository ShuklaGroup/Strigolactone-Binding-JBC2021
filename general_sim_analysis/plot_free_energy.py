"""Plot free energy on given coordinates"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import glob
plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

def raw_plot(data1, data2, label1="Label Your Axes!", label2="Label Your Axes!", savename="fep.png"):

    plt.figure()
    plt.hexbin(data1, data2, bins='log', mincnt=1, cmap='jet', vmin=0, vmax=3.0, extent=(0,6,0,1.4), gridsize=300)
    cbar = plt.colorbar()
    cbar.set_label("Probability Density")
    plt.tick_params(axis='both',labelsize=14)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.xlim(0,6)
    plt.ylim(0,1.4)
    axes = plt.axes()
    axes.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    plt.grid(linestyle=":")
    plt.savefig(savename)

def get_prob_density(data1, data2, bins=300, binrange=[[0,1],[0,4]], weights=None):

    if len(np.shape(data1)) > 1:
        data1 = np.asarray(data1)[:,0]
        data2 = np.asarray(data2)[:,0]

    hist, x_edges, y_edges = np.histogram2d(data1, data2, bins=300, range=binrange, weights=weights)

    prob_density = hist/np.sum(hist)
    x_coords = 0.5*(x_edges[:-1]+x_edges[1:]) 
    y_coords = 0.5*(y_edges[:-1]+y_edges[1:])

    return prob_density.T, x_coords, y_coords

def free_energy(data1, data2, T=300, weights=None, lims=(0,4,0,1), max_energy=6, label1="Label Your Axes!", label2="Label Your Axes!", savename="fe.png"):

    #compute free energy
    gas_constant = 0.00198588
    prob, x, y = get_prob_density(data1, data2, binrange=[[lims[0],lims[1]],[lims[2],lims[3]]], weights=weights)
    X, Y = np.meshgrid(x,y)
    free_energy = -gas_constant*T*np.log(prob)
    free_energy -= np.min(free_energy)
    
    plt.figure()
    fig, ax = plt.subplots()
    plt.contourf(X, Y, free_energy, np.linspace(0, max_energy, max_energy*5+1), vmin=0.0, vmax=max_energy, cmap='jet')
    cbar = plt.colorbar(ticks=range(max_energy+1))
    cbar.set_label("Free Energy (kcal/mol)",size=24)
    cbar.ax.set_yticklabels(range(max_energy+1))
    cbar.ax.tick_params(labelsize=20)
    plt.tick_params(axis='both',labelsize=20)
    plt.xlabel(label1)
    plt.ylabel(label2)

    if lims[1] - lims[0] <= 1:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    elif lims[1] - lims[0] <= 2:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    elif lims[1] - lims[0] > 100:
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))

    if lims[3] - lims[2] <= 1:
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    elif lims[3] - lims[2] <= 2:
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    elif lims[3] - lims[2] > 100:
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))

    plt.xlim(lims[0],lims[1])
    plt.ylim(lims[2],lims[3])
    plt.grid(linestyle=":")
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.gca().set_aspect(aspect=(lims[1]-lims[0])/(lims[3]-lims[2]), adjustable='box')
    fig.tight_layout()
    plt.savefig(savename, transparent=True)
    plt.close()

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", nargs=1, type=str, default="analysis", help="Directory containing featurization files")
    parser.add_argument("--ax_files", nargs=2, type=str, help="Identifier for files")
    parser.add_argument("--ax_labels", nargs=2, type=str, help="Axis labels")
    parser.add_argument("--ax_lims", nargs=4, type=float, help="Axis limits")
    parser.add_argument("--max_energy", nargs=1, default=6, type=int, help="Maximum free energy")
    parser.add_argument("--weights", type=str, help="Pickle file containing MSM weights")
    parser.add_argument("--savename", type=str, help="Output file name")

    args = parser.parse_args()
    
    return args

if __name__=="__main__":

    args = get_args()

    ax1_files = sorted(glob.glob("%s/*%s*"%(args.feat_dir, args.ax_files[0])))
    ax2_files = sorted(glob.glob("%s/*%s*"%(args.feat_dir, args.ax_files[1])))

    ax1_data = []
    ax2_data = []

    for data_file in ax1_files:
        ax1_data.extend(np.load(data_file))

    ax1_data = np.array(ax1_data)

    if len(np.shape(ax1_data)) > 1:
        ax1_data = ax1_data[:,0]

    for data_file in ax2_files:
        ax2_data.extend(np.load(data_file))

    ax2_data = np.array(ax2_data)    

    if len(np.shape(ax2_data)) > 1:
        ax2_data = ax2_data[:,0]

    print(np.shape(ax1_data))
    print(np.shape(ax2_data))

    if args.weights == None:
        free_energy(ax1_data, ax2_data, lims=(args.ax_lims[0], args.ax_lims[1], args.ax_lims[2], args.ax_lims[3]), max_energy=args.max_energy, label1=args.ax_labels[0], label2=args.ax_labels[1], savename=args.savename)

    else:
        import pickle
        weights_all = pickle.load(open(args.weights,'rb'))
        weights = []
        for traj in weights_all:
            weights.extend(traj)
        weights = np.array(weights)
        print(np.shape(weights))
        free_energy(ax1_data, ax2_data, lims=(args.ax_lims[0], args.ax_lims[1], args.ax_lims[2], args.ax_lims[3]), max_energy=args.max_energy, label1=args.ax_labels[0], label2=args.ax_labels[1], savename=args.savename, weights=weights)
