import mdtraj as md
import glob
import numpy as np

#targettopfile="./stripped.4ih4_withlig_noh.prmtop"
targettopfile="./stripped.ShHTL7_withGR24.prmtop"

for file in glob.glob('./*.xtc'):
    t = md.load(file, top=targettopfile)

    count=1
    for res in range(211, 218): #AtD14
    #for res in range(215, 222): #ShHTL7
        d = md.compute_contacts(t,[[res,242]], scheme='ca')[0] # Loop res-H distance (AtD14)
        #d = md.compute_contacts(t,[[res,245]], scheme='ca')[0] # Loop res-H distance (ShHTL7)
        n_frames = t.n_frames

        dis = np.empty([n_frames, 1])

        for i in range(n_frames):
          dis[i,0:1]=d[i][0]

        np.save('./analysis/'+file.split('/')[-1]+'_ftr-loop-distance-%d.npy'%count,dis)
        count+=1
