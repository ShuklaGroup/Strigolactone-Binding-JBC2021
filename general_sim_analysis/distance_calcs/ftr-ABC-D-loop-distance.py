import glob
import numpy as np
import mdtraj as md

targettopfile="./stripped.4ih4_withlig_noh.prmtop"
#targettopfile="./stripped.ShHTL7_withGR24.prmtop"

d_loop_ca = [3321, 3349, 3384] #AtD14
#d_loop_ca = [3330, 3361, 3402] #ShHTL7

for file in glob.glob('./stripped/*nc'):
    t = md.load(file, top=targettopfile)


    for j in range(len(d_loop_ca)):

        d = md.compute_distances(t,[[d_loop_ca[j],4070]]) #Ligand C4-Ser97 OG (D14 with GR24)
        #d = md.compute_distances(t,[[d_loop_ca[j],4156]]) #Ligand C4-Ser97 OG (HTL7)
        n_frames = t.n_frames

        dis = np.empty([n_frames, 1])

        for i in range(n_frames):
          dis[i,0:1]=d[i][0]

        np.save('./analysis/'+file.split('/')[-1]+'_ftr-ABC-D-loop-distance-%d.npy'%(j+1), dis)
