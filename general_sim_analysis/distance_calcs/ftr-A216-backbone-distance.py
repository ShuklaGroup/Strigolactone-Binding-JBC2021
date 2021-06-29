import mdtraj as md
import glob
import numpy as np

targettopfile="./stripped.4ih4_withlig_noh.prmtop"
#targettopfile="./stripped.ShHTL7_withGR24.prmtop"

bbone_res = [237, 239] #AtD14
#bbone_res = [240, 242] #ShHTL7

for file in glob.glob('./stripped/*nc'):
    t = md.load(file, top=targettopfile)

    for j in range(len(bbone_res)):

        d = md.compute_contacts(t,[[211,bbone_res[j]]], scheme='closest-heavy')[0]     	# distance between the two residues on helix T3 and T1 respectively
        #d = md.compute_contacts(t,[[214,bbone_res[j]]], scheme='closest-heavy')[0]     	# distance between the two residues on helix T3 and T1 respectively (ShHTL7)
        n_frames = t.n_frames

        dis = np.empty([n_frames, 1])

        for i in range(n_frames):
          dis[i,0:1]=d[i][0]

        np.save('./analysis/'+file.split('/')[-1]+'_ftr-A216-backbone-distance-%d.npy'%(j+1),dis)
