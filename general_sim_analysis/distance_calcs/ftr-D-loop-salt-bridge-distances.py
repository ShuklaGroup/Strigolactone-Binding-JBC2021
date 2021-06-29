import glob
import numpy as np
import mdtraj as md

#targettopfile="stripped.4ih4_withlig_noh.prmtop"
targettopfile="stripped.ShHTL7_withGR24.prmtop"

#pairs = [[213,238],[212,240],[212,162]] #AtD14
pairs = [[216,241],[215,243],[215,164]] #ShHTL7
labels = [1,2,3]

for file in glob.glob('./stripped/*.xtc'):
    t = md.load(file, top=targettopfile)
    for j in range(len(labels)):

        d = md.compute_contacts(t, [pairs[j]], scheme='sidechain-heavy')[0]
        n_frames = t.n_frames

        dis = np.empty([n_frames, 1])

        for i in range(n_frames):
          dis[i,0:1]=d[i][0]

        np.save('./analysis/'+file.split('/')[-1]+'_ftr-D-loop-salt-bridge-distance-%d.npy'%labels[j], dis)
