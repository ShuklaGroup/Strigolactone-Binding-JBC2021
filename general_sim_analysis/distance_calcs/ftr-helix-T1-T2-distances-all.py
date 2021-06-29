import glob
import numpy as np
import mdtraj as md

#targettopfile="stripped.4ih4_withlig_noh.prmtop"
#targettopfile="stripped.ShHTL7_withGR24.prmtop"

pairs = [[135,157],[138,153],[142,149]] #AtD14
pairs = [[137,159],[140,155],[144,151]] #ShHTL7
labels = ["1","2","3"]

for file in glob.glob('./*.xtc'):
    t = md.load(file, top=targettopfile)
    for j in range(len(labels)):

        d = md.compute_contacts(t, [pairs[j]], scheme='ca')[0]     	# distance between residue pairs on helix T1 and T2 respectively
        n_frames = t.n_frames

        dis = np.empty([n_frames, 1])

        for i in range(n_frames):
          dis[i,0:1]=d[i][0]

        np.save('./analysis/'+file.split('/')[-1]+'_ftr-helix-T1-T2-distance-%s.npy'%labels[j], dis)
