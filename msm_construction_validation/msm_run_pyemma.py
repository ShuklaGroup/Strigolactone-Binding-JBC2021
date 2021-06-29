"""Construct MSM"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyemma
import glob
import pickle
import time

localtime = time.asctime(time.localtime(time.time()))
print("Began run at: ",localtime)

#Read in data, combine features into single matrix
feat_identifiers = []
feat_identifiers.extend(["*ABC*pocket*","*D-ring*pocket*"]) #Ligand-pocket distances
feat_identifiers.extend(["*T1-T2*top*", "*T1-T2*bot*", "*T1-T2*mid*"]) #T1-T2 distances
feat_identifiers.extend(["*ABC*T1*top*", "*ABC*T1*mid*", "*ABC*T1*bot*"]) #A-ring-T1 distances
feat_identifiers.extend(["*ABC*T2*top*", "*ABC*T2*mid*", "*ABC*T2*bot*"]) #A-ring-T2 distances
feat_identifiers.extend(["*D-ring*T1*top*", "*D-ring*T1*mid*", "*D-ring*T1*bot*"]) #D-ring-T1 distances
feat_identifiers.extend(["*D-ring*T2*top*", "*D-ring*T2*mid*", "*D-ring*T2*bot*"]) #D-ring-T2 distances
feat_identifiers.extend(["*ftr-loop-dist*1*", "*ftr-loop-dist*2*", "*ftr-loop-dist*3*", "*ftr-loop-dist*4*", "*ftr-loop-dist*5*", "*ftr-loop-dist*6*", "*ftr-loop-dist*7*"]) #Loop distances
feat_identifiers.extend(["*ABC-D-loop-dist*1*", "*ABC-D-loop-dist*2*", "*ABC-D-loop-dist*3*", "*D-ring-D-loop-dist*1*", "*D-ring-D-loop-dist*2*", "*D-ring-D-loop-dist*3*"]) #Ligand-loop distances

file_lists = []

params_used = open("Parameters.txt",'w')
for ft in feat_identifiers:
    params_used.write("%s \n"%ft)
    file_lists.append(sorted(glob.glob("../analysis/%s"%ft)))

feat_all = []
for i in range(len(file_lists[0])): #Iterate through features
    feat = [] #Initialize feature for j-th trajectory
    for j in range(len(file_lists)):
        feat.append(np.load(file_lists[j][i])) #Load i-th feature for j-th trajectory
    feat_all.append(np.hstack(feat))

print(len(feat_all))
pickle.dump(feat_all, open("lig_pocket_dist.pkl",'wb'))

##MSM Parameters
lag_time = 286 #Trajectory steps
tica_components = 4
n_clusters = 325

params_used.write("Lag time: %d \n"%lag_time)
params_used.write("TICA components: %d \n"%tica_components)
params_used.write("Clusters: %d"%n_clusters)
params_used.close()

localtime = time.asctime(time.localtime(time.time()))
print("Began TICA at: ",localtime)

tica_object = pyemma.coordinates.tica(data=feat_all, lag=4, dim=tica_components)
tica_trajs = tica_object.get_output()
pickle.dump(tica_object, open("tica_object.pkl",'wb'))
pickle.dump(tica_trajs, open("tica_trajs.pkl",'wb'))

localtime = time.asctime(time.localtime(time.time()))
print("Began clustering at: ",localtime)

cluster_object = pyemma.coordinates.cluster_mini_batch_kmeans(tica_trajs, max_iter=200, k=n_clusters)
dtrajs = cluster_object.assign(tica_trajs)
pickle.dump(dtrajs, open("dtrajs.pkl",'wb'))

its_object = pyemma.msm.its(dtrajs, errors='bayes', lags=800, nits=10)
pyemma.plots.plot_implied_timescales(its_object, outfile="its_plot.png", units='ns', dt=0.07)

localtime = time.asctime(time.localtime(time.time()))
print("Began estimating MSM at: ",localtime)

msm_object = pyemma.msm.estimate_markov_model(dtrajs, lag_time, dt_traj='70 ps', score_method='VAMP1', score_k=10)
pickle.dump(msm_object, open("msm_object.pkl",'wb'))

localtime = time.asctime(time.localtime(time.time()))
print("Data usage: ", msm_object.active_count_fraction)

print("Began scoring at: ",localtime)
score = msm_object.score_cv(dtrajs, score_method='VAMP1', score_k=5)
print(score)

localtime = time.asctime(time.localtime(time.time()))
print("Finished scoring at: ",localtime)

weights = msm_object.trajectory_weights()
pickle.dump(weights, open("msm_weights.pkl",'wb'))

eigs = msm_object.eigenvalues()
plt.figure()
plt.plot(list(range(len(eigs))), eigs, 'o')
plt.savefig("eigenvalues.png")

cktest = msm_object.cktest(10, mlags=range(4))
fig, ax = pyemma.plots.plot_cktest(cktest)
fig.savefig("cktest.png")
