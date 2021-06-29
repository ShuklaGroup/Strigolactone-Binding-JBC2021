import glob
import numpy as np
import mdtraj as md

targettopfile="./stripped.4ih4_withlig_noh.prmtop"
#targettopfile="./stripped.ShHTL7_withGR24.prmtop"

#res_ca = [2349, 2405, 2443] #ShHTL7
res_ca = [2334, 2398, 2442] #AtD14
labels = ['top','mid','bot']

for file in glob.glob('./stripped/*nc'):
    t = md.load(file, top=targettopfile)
    #d = md.compute_distances(t,[[2442,4070]]) #Ligand C4-Ser97 OG (D14)

    for j in range(len(labels)):
        #d = md.compute_distances(t,[[res_ca[j],4173]]) #ShHTL7
        d = md.compute_distances(t,[[res_ca[j],4087]]) #AtD14 with GR24
        n_frames = t.n_frames

        dis = np.empty([n_frames, 1])

        for i in range(n_frames):
          dis[i,0:1]=d[i][0]

        np.save('./analysis/'+file.split('/')[-1]+'_ftr-D-ring-T2-distance-%s.npy'%labels[j] ,dis)
