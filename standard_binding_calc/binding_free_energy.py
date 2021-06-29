import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

import pickle
import glob
import argparse
import os

global GAS_CONSTANT
GAS_CONSTANT = 0.00198588 #Gas constant

def get_lig_xyz(traj_dir, topfile, atom_ind, generate=True):
    """Get ligand atom coordinates from aligned trajectories"""

    if generate:
        aligned_trajs = glob.glob("%s/*"%traj_dir)

        if not os.path.isdir("./xyz_projection"):
            os.mkdir("./xyz_projection")

        for traj in aligned_trajs:
            xyz_traj = md.load(traj, top=topfile).xyz[:,atom_ind,:]
            np.save('./xyz_projection/'+traj.split('/')[-1]+'_ftr-xyz.npy', xyz_traj)

    #Load xyz projection if already exists:
    xyz_files = sorted(glob.glob("xyz_projection/*"))
    lig_xyz = []
    for xyz in xyz_files:
        lig_xyz.append(np.load(xyz))

    lig_xyz = np.concatenate(lig_xyz)

    return lig_xyz

def get_prob_density_2d(data1, data2, bins=300, binrange=None, weights=None):
    """Gets 2D probability density for visualization purposes. Do not use for volume correction!"""

    if len(np.shape(data1)) > 1:
        data1 = np.asarray(data1)[:,0]
        data2 = np.asarray(data2)[:,0]

    prob_density, x_edges, y_edges = np.histogram2d(data1, data2, bins=bins, density=True, range=binrange, weights=weights)

    x_coords = 0.5*(x_edges[:-1]+x_edges[1:])
    y_coords = 0.5*(y_edges[:-1]+y_edges[1:])

    return prob_density, x_coords, y_coords

def get_prob_density_3d(data, bins=300, binrange=None, weights=None):
    """NOTE: Input data for this as one single array!!"""

    prob_density, edges = np.histogramdd(data, bins=bins, density=True, range=binrange, weights=weights)
    
    x_coords = 0.5*(edges[0][:-1]+edges[0][1:])
    y_coords = 0.5*(edges[1][:-1]+edges[1][1:])
    z_coords = 0.5*(edges[2][:-1]+edges[2][1:])

    return prob_density, x_coords, y_coords, z_coords

def get_pmf_3d(data, T=300, bins=300, weights=None, lims=(-7,7,-7,7,-7,7)):
    """MSM-weighted 3D free energy landscape"""

    prob, x, y, z = get_prob_density_3d(data, bins=bins, binrange=[[lims[0],lims[1]],[lims[2],lims[3]],[lims[4],lims[5]]], weights=weights)
    X, Y, Z = np.meshgrid(x,y,z)
    free_energy = -GAS_CONSTANT*T*np.log(prob)
    free_energy -= np.min(free_energy) 

    return free_energy, X, Y, Z

def plot_2d(data1, data2, T=300, weights=None, lims=(0,4,0,1), max_energy=6, label1="Label Your Axes!", label2="Label Your Axes!", savename="fe.png"):
    """Plot 2D landscape. Modified from other landscape plotter"""

    prob, x, y = get_prob_density_2d(data1, data2, binrange=[[lims[0],lims[1]],[lims[2],lims[3]]], weights=weights)
    X, Y = np.meshgrid(x,y)
    free_energy = -GAS_CONSTANT*T*np.log(prob)
    free_energy -= np.min(free_energy)

    min_point = np.where(free_energy==0)

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

    plt.gca().set_aspect(aspect=(lims[1]-lims[0])/(lims[3]-lims[2]), adjustable='box')
    fig.tight_layout()
    plt.savefig(savename, transparent=True)

def calc_Vb(pmf, X, Y, Z, T=300, spatial_cutoff=1.0, fe_cutoff=1.0):
    """Integrate over bound volume. Assumes that the global minimum on the pmf is in the bound minimum."""

    #Find minimum point on 3D PMF. USE MSM WEIGHTED PMF ONLY.
    min_point = np.array((X0, Y0, Z0))
    print(min_point)
       
    #Get points with free energy < fe_cutoff
    pmf_bound = pmf[pmf < fe_cutoff]
    x_bound = X[pmf < fe_cutoff]
    y_bound = Y[pmf < fe_cutoff]
    z_bound = Z[pmf < fe_cutoff]

    coords_bound = np.vstack((x_bound, y_bound, z_bound)).T

    #Remove points farther than spatial_cutoff from minimum point (X0, Y0, Z0)
    far_points = [] #Initialize list of points to remove
    for i in range(len(pmf_bound)):
        if np.linalg.norm(coords_bound[i,:] - min_point) > spatial_cutoff:
            far_points.append(i)

    pmf_points_use = np.delete(pmf_bound, far_points)
    coords_use = np.delete(coords_bound, far_points, 0)

    #Compute integral
    dV = (X[0,1,0]-X[0,0,0])*(Y[1,0,0]-Y[0,0,0])*(Z[0,0,1]-Z[0,0,0])
    Vb = dV*np.sum(np.exp(-(1.0/(GAS_CONSTANT*T))*pmf_points_use))
    
    return Vb

def calc_dG_binding(pmf, Vb, V0=1.661, T=300):

    pmf = np.ma.masked_where(pmf > 100, pmf)
    dG_binding = -GAS_CONSTANT*T*np.log(Vb/V0) - np.max(pmf)

    return dG_binding

def sensitivity_test(pmf, X, Y, Z, T=300, spatial_cutoff_range=np.linspace(0.2,1.0,9), fe_cutoff_range=np.linspace(0.5,5,28)):
    """Computes and plots Vb and dG using a range of spatial and free energy cutoff values"""
    
    Vb_all = np.zeros((np.size(spatial_cutoff_range), np.size(fe_cutoff_range)))
    dG_all = np.zeros((np.size(spatial_cutoff_range), np.size(fe_cutoff_range)))

    for i in range(np.size(spatial_cutoff_range)):
        for j in range(np.size(fe_cutoff_range)):
            Vb_all[i,j] = calc_Vb(pmf, X, Y, Z, T=T, spatial_cutoff=spatial_cutoff_range[i], fe_cutoff=fe_cutoff_range[j])
            dG_all[i,j] = calc_dG_binding(pmf, Vb_all[i,j], T=T)

    #Plot Vb
    plt.figure()
    fig, ax = plt.subplots()
    for i in range(np.size(spatial_cutoff_range)):
        plt.plot(fe_cutoff_range, Vb_all[i,:], label="Cutoff=%0.1f nm"%spatial_cutoff_range[i])
    #plt.legend(fontsize=8)
    plt.xlabel("Free Energy Cutoff (kcal/mol)")
    plt.ylabel("V$_b$ (nm$^3$)")
    plt.xlim(0,5)
    plt.ylim(0.0,0.1)
    fig.tight_layout()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.savefig("Vb_sensitivity.png", transparent=True)
    plt.close()
    
    #Plot dG
    plt.figure()
    fig, ax = plt.subplots()
    for i in range(np.size(spatial_cutoff_range)):
        plt.plot(fe_cutoff_range, dG_all[i,:], label="Cutoff=%0.1f nm"%spatial_cutoff_range[i])
    #plt.legend(fontsize=8)
    plt.xlabel("Free Energy Cutoff (kcal/mol)")
    plt.ylabel("$\Delta$G (kcal/mol)")
    plt.xlim(0,5)
    plt.ylim(-7,-3)
    fig.tight_layout()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.savefig("dG_sensitivity.png", transparent=True)
    plt.close()
 
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_name", type=str, nargs=1, default="lig_binding", help="Name for system")
    parser.add_argument("--traj_path", type=str, nargs=1, default="xtc_trajs", help="Path containing aligned trajectories")
    parser.add_argument("--top_file", type=str, nargs=1, default="../stripped.4ih4_withlig_noh.prmtop", help="Path containing topology")
    parser.add_argument("--generate_xyz", type=bool, nargs=1, default=False, help="Set to True to generate ligand xyz coordinates")
    parser.add_argument("--lig_atom", type=int, nargs=1, default=4087, help="Atom index of ligand atom")
    parser.add_argument("--msm_weights", type=str, nargs=1, default="../msm_final_fixed/msm_weights.pkl", help="Path to MSM weights file")
    parser.add_argument("--ax_lims", type=int, nargs=2, default=(-7,7), help="Minimum and maximum axis values. Same in all 3 dimensions")
    parser.add_argument("--spatial_cutoff", type=float, nargs=1, default=1.0, help="Spatial cutoff distance for definition of bound state")
    parser.add_argument("--fe_cutoff", type=float, nargs=1, default=1.0, help="Free energy cutoff for definition of bound state")
    parser.add_argument("--sensitivity_test", type=bool, nargs=1, default=False, help="Indicate whether to run a sensitivity test on spatial and free energy cutoffs")
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args=get_args()

    if type(args.generate_xyz) == 'list':
        lig_xyz = get_lig_xyz(args.traj_path[0], args.top_file[0], args.lig_atom[0], generate=args.generate_xyz[0])
    else:
        lig_xyz = get_lig_xyz(args.traj_path[0], args.top_file[0], args.lig_atom[0], generate=args.generate_xyz)

    weights_all=pickle.load(open(args.msm_weights[0], 'rb'))
    weights = []
    for traj in weights_all:
        weights.extend(traj)
    weights = np.array(weights)

    #Plot x-y
    plot_2d(lig_xyz[:,0], lig_xyz[:,1], T=300, weights=None, lims=args.ax_lims*2, max_energy=7, label1="x (nm)", label2="y (nm)", savename="%s_xy_raw.png"%args.sys_name[0])
    plot_2d(lig_xyz[:,1], lig_xyz[:,2], T=300, weights=None, lims=args.ax_lims*2, max_energy=7, label1="y (nm)", label2="z (nm)", savename="%s_yz_raw.png"%args.sys_name[0])
    plot_2d(lig_xyz[:,0], lig_xyz[:,2], T=300, weights=None, lims=args.ax_lims*2, max_energy=7, label1="x (nm)", label2="z (nm)", savename="%s_xz_raw.png"%args.sys_name[0])

    plot_2d(lig_xyz[:,0], lig_xyz[:,1], T=300, weights=weights, lims=args.ax_lims*2, max_energy=7, label1="x (nm)", label2="y (nm)", savename="%s_xy_weighted.png"%args.sys_name[0])
    plot_2d(lig_xyz[:,1], lig_xyz[:,2], T=300, weights=weights, lims=args.ax_lims*2, max_energy=7, label1="y (nm)", label2="z (nm)", savename="%s_yz_weighted.png"%args.sys_name[0])
    plot_2d(lig_xyz[:,0], lig_xyz[:,2], T=300, weights=weights, lims=args.ax_lims*2, max_energy=7, label1="x (nm)", label2="z (nm)", savename="%s_xz_weighted.png"%args.sys_name[0])

    #3-D histogram
    pmf_3d, X, Y, Z = get_pmf_3d(lig_xyz, lims=args.ax_lims*3, weights=weights)
    Vb = calc_Vb(pmf_3d, X, Y, Z, spatial_cutoff=args.spatial_cutoff[0], fe_cutoff=args.fe_cutoff[0])
    dG_binding = calc_dG_binding(pmf_3d, Vb)

    print("STANDARD BINDING FREE ENERGY (kcal/mol)")
    print(dG_binding)

    if args.sensitivity_test[0]:
        sensitivity_test(pmf_3d, X, Y, Z)
