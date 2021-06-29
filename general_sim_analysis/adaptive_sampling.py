"""Cluster simulation data on given metrics and seed new simulations"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn.cluster as cls
import mdtraj as md

from itertools import cycle, islice

import argparse
import pickle
import glob
import re

cls_methods = {"MiniBatchKMeans":cls.MiniBatchKMeans, "KMeans":cls.KMeans}

def load_data(metrics_list, dtype="glob", ndim=2):
    """Returns (n_frames)x(n_metrics) numpy array with metrics on which to cluster"""

    data = []
    fnames = []
    frameid = []

    if dtype == "glob":
        for metric in metrics_list:
            files = sorted(glob.glob(metric))
            single_data = []
            single_fnames = []
            single_frameid = []
            for f in files:
                try:
                    ftr = np.load(f)[:,0]
                except:
                    ftr = np.load(f)
                traj_len = np.size(ftr)
                single_data.extend(ftr)
                single_fnames.extend(traj_len*[f[:f.find("_ftr")]])
                single_frameid.extend(range(traj_len))
            data.append(single_data)
            fnames.append(single_fnames)
            frameid.append(single_frameid)
        data = np.array(data).T

    elif dtype == "pickle":
        ftr_trajs = pickle.load(open("%s.pkl"%metrics_list[0], 'rb'))
        traj_files = sorted(glob.glob("stripped/*"))

        #Change file names for consistency with other data loader
        for i in range(len(traj_files)):
            traj_files[i] = re.sub("stripped", "analysis", traj_files[i])

        for dim in range(ndim):
            single_data = []
            single_fnames = []
            single_frameid = []
            for i in range(len(ftr_trajs)):
                traj_len = np.shape(ftr_trajs[i])[0]
                single_data.extend(ftr_trajs[i][:,dim])
                single_fnames.extend(traj_len*[traj_files[i]])
                single_frameid.extend(range(traj_len))
            data.append(single_data)
            fnames.append(single_fnames)
            frameid.append(single_frameid)
        data = np.array(data).T

    else:
        print("File loading for input type not implemented!")
        pass

    return data, fnames[0], frameid[0]

def filter_data(data, fnames, frameid, cutoffs):
    """Only use a subset of the input data for clustering. Cutoffs is a list of 2x1 arrays"""
    
    data_use = data[:]
    fnames_use = fnames[:]
    frameid_use = frameid[:]

    for i in range(np.shape(data_use)[1]):
        delete_frames = []
        for j in range(np.shape(data_use)[0]):
            if data_use[j,i] < cutoffs[i,0] or data_use[j,i] > cutoffs[i,1]:
                delete_frames.append(j)
        data_use = np.delete(data_use, delete_frames, axis=0)
        fnames_use = np.delete(fnames_use, delete_frames, axis=0)
        frameid_use = np.delete(frameid_use, delete_frames, axis=0)

    return data_use, fnames_use, frameid_use

def cluster(dataset, method="MiniBatchKMeans", cls_kwargs={"n_clusters":5}):

    cls_object = cls_methods[method](**cls_kwargs)
    clusters = cls_object.fit_predict(dataset)

    n_clusters = cls_kwargs["n_clusters"]
    cluster_pop = np.zeros(n_clusters)

    for c in clusters:
        cluster_pop[c] += 1

    return clusters, cluster_pop, cls_object.cluster_centers_

def find_least_count_clusters(clusters, cluster_pop, n=10):

    lc_clusters = np.argsort(cluster_pop)[:n]
    lc_cluster_pop = cluster_pop[lc_clusters]

    return lc_clusters, lc_cluster_pop

def plot_clusters(dataset, clusters, cluster_centers, label1="Label Your Axes!", label2="Label Your Axes!", savename="clusters.png"):
    
    plt.figure()
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']), int(max(clusters) + 1))))

    plt.scatter(dataset[:,0], dataset[:,1], s=2, color=colors[clusters])

    cluster_labels = list(range(np.shape(cluster_centers)[0]))
    
    for label, x, y in zip(cluster_labels, cluster_centers[:,0], cluster_centers[:,1]):
        plt.annotate(label, xy=(x,y))

    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.savefig("%s"%savename)

def draw_samples(fnames, frameid, clusters, clusters_to_sample=None, n_per_cluster=5):
    """Randomly picks n_per_cluster samples from each cluster in clusters_to_sample"""
    
    if clusters_to_sample is None:
        clusters_to_sample = list(range(np.max(clusters)+1)) #Draw samples from all clusters if none specified
    
    fnames_smp = []
    frameid_smp = []
    clusterid_smp = []

    for c in clusters_to_sample:
        indices = np.where(clusters==c)[0] #Get indices for data points in cluster
        if len(indices) <= n_per_cluster:
            frames_to_pick = list(range(len(indices)))
        else:
            frames_to_pick = np.random.randint(len(indices), size=n_per_cluster)

        for f in frames_to_pick:
            ind = indices[f]

            fnames_smp.append(fnames[ind])
            frameid_smp.append(frameid[ind])
            clusterid_smp.append(clusters[ind])

    return fnames_smp, frameid_smp, clusterid_smp

def get_rst(fnames, frameid, clusterid, topfile):
    """Saves rst files. Topfile with full system required."""
    for i in range(len(fnames)):
        fname = fnames[i]
        if "holo" in fname: #TEMPORARY HACK FOR D14 BINDING DATASET
            traj = re.sub("analysis", "/home/jiming/Storage/AtD14_holo/trajectories_unstripped", fname)
        elif "binding-path" in fname:
            traj = re.sub("analysis", "/home/jiming/Storage/AtD14_binding_path/trajectories", fname)
        else:
            traj = re.sub("analysis","trajectories", fname)

        traj = re.sub(".strip", "", traj)
        #traj = re.sub(".nc", ".mdcrd", traj)

        try:
            f = md.load_frame(traj, frameid[i], top=topfile)
        except:
            f = md.load_netcdf(traj, top=topfile, frame=frameid[i])

        if "mdcrd" in traj:
            savename = re.sub(".mdcrd", "", traj)
        else:
            savename = re.sub(".nc", "", traj)

        f.save_amberrst7("cls_%d_fr_%d_traj_%s.rst"%(clusterid[i], frameid[i], savename[-5:]))

def get_pdbs(fnames, frameid, clusterid, topfile):
    """Saves pdb files. Defaults to stripped system."""
    for i in range(len(fnames)):
        fname = fnames[i]
        traj = re.sub("analysis", "stripped", fname)
        try:
            f = md.load_frame(traj, frameid[i], top=topfile)
        except:
            traj = re.sub(".mdcrd", ".nc", traj)
            f = md.load_frame(traj, frameid[i], top=topfile)

        savename = re.sub(".strip.nc", "", traj)
        f.save_pdb("cls_%d_fr_%d_traj_%s.pdb"%(clusterid[i], frameid[i], savename[-5:]))

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ftr_dir", type=str, default='analysis', help="Directory containing featurizations")
    parser.add_argument("--metric_ids", type=str, nargs='+')
    parser.add_argument("--input_type", type=str, default='glob')
    parser.add_argument("--cutoffs", type=float, nargs='+')
    parser.add_argument("--n_clusters", type=int, default=40)
    parser.add_argument("--n_per_cluster", type=int, default=10)
    parser.add_argument("--lc", type=int, help="Draw samples from --lc least count clusters")
    parser.add_argument("--topfile", type=str)
    parser.add_argument("--get_pdbs", action='store_true')
    parser.add_argument("--get_rst", action='store_true')
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()
    metrics = []

    if args.input_type == "glob":
        for m in args.metric_ids:
            metrics.append("%s/*%s*"%(args.ftr_dir, m))
    else:
        metrics.append("%s/%s"%(args.ftr_dir, args.metric_ids[0]))

    data, fnames, frameid = load_data(metrics, dtype=args.input_type)

    if args.cutoffs is not None:
        cutoffs = np.reshape(np.array(args.cutoffs), (int(len(args.cutoffs)/2), 2))
        data, fnames, frameid = filter_data(data, fnames, frameid, cutoffs)

    print(np.shape(data))
    clusters, cluster_pop, cluster_centers = cluster(data, cls_kwargs={"n_clusters":args.n_clusters})

    #Find least populated states and draw samples
    if args.lc is not None:
        lc_clusters, lc_cluster_pop = find_least_count_clusters(clusters, cluster_pop, n=args.lc)
        print(lc_clusters)
        print(lc_cluster_pop)

        fnames_smp, frameid_smp, clusterid_smp = draw_samples(fnames, frameid, clusters, clusters_to_sample=lc_clusters, n_per_cluster=args.n_per_cluster)
        pickle.dump(fnames_smp, open("fnames_lc.pkl", "wb"), protocol=2)
        pickle.dump(frameid_smp, open("frameid_lc.pkl", "wb"), protocol=2)
        pickle.dump(clusterid_smp, open("clusterid_lc.pkl", "wb"), protocol=2)
    else:
        #Draw samples from all the states
        fnames_smp, frameid_smp, clusterid_smp = draw_samples(fnames, frameid, clusters, n_per_cluster=args.n_per_cluster)

        pickle.dump(fnames_smp, open("fnames_all.pkl", "wb"), protocol=2)
        pickle.dump(frameid_smp, open("frameid_all.pkl", "wb"), protocol=2)
        pickle.dump(clusterid_smp, open("clusterid_all.pkl", "wb"), protocol=2)

    plot_clusters(data, clusters, cluster_centers, label1="Metric 1", label2="Metric 2", savename="clusters.png")
    
    if args.get_pdbs:
        get_pdbs(fnames_smp, frameid_smp, clusterid_smp, "stripped.%s"%args.topfile)

    if args.get_rst:
        get_rst(fnames_smp, frameid_smp, clusterid_smp, args.topfile)

