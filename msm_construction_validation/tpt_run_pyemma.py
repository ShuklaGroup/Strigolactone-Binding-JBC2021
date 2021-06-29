"""Transition path theory to estimate fluxes between major states found in an MD dataset"""

import numpy as np
import mdtraj as md
import pyemma
import pickle
import glob

def get_samples(msm_obj, topfile, n=1000):
    """Draw samples from each MSM state"""

    samples = msm_obj.sample_by_state(n)
    trajs = sorted(glob.glob("../stripped/*"))
    #Save xtc files
    for i in range(len(samples)):
        state_samples = samples[i]
        state_traj = []
        for j in range(len(state_samples)):
            state_traj.append(md.load_frame(trajs[state_samples[j,0]], state_samples[j,1], top=topfile))
        md.join(state_traj).save_xtc("state_%d.xtc"%i)

    return samples

def id_bound_unbound_states(samples):
    """Identify ligand bound, unbound, and inverse bound states"""

    ABC_pocket_dist = sorted(glob.glob("../analysis/*ABC-pocket*")) #Ligand pocket features
    D_ring_pocket_dist = sorted(glob.glob("../analysis/*D-ring-pocket*"))

    bound_states = []
    unbound_states = []
    inverse_bound_states = []

    for i in range(len(samples)):
        state_samples = samples[i]
        abc_dist = []
        d_dist = []
        for j in range(len(state_samples)):
            abc_dist.append(np.load(ABC_pocket_dist[state_samples[j,0]])[state_samples[j,1]])
            d_dist.append(np.load(D_ring_pocket_dist[state_samples[j,0]])[state_samples[j,1]])
        if all((np.min(d_dist) < 0.6, np.mean(abc_dist) > np.mean(d_dist))):
            bound_states.append(i)
        elif np.mean(abc_dist) > 1.5 and np.mean(d_dist) > 1.5:
            unbound_states.append(i)
        elif all((np.min(abc_dist) < 1.5, np.min(d_dist) < 1.5, np.mean(abc_dist) < np.mean(d_dist))):
            inverse_bound_states.append(i)

        inverse_bound_states, _, _ = trim_state_sets(inverse_bound_states, bound_states)

    return bound_states, unbound_states, inverse_bound_states

def id_anchored_states(samples):
    """Identify states where ligand is anchored to lid helices"""

    A_ring_T1_dist_top = sorted(glob.glob("../analysis/*ABC*T1*top*"))
    A_ring_T2_dist_top = sorted(glob.glob("../analysis/*ABC*T2*top*"))
    D_ring_T1_dist_bot = sorted(glob.glob("../analysis/*D-ring*T1*bot*"))
    D_ring_T2_dist_bot = sorted(glob.glob("../analysis/*D-ring*T2*bot*"))

    anchored_states = []

    for i in range(len(samples)):
        state_samples = samples[i]
        abc_t1_top = []
        abc_t2_top = []
        d_t1_bot = []
        d_t2_bot = []
        for j in range(len(state_samples)):
            abc_t1_top.append(np.load(A_ring_T1_dist_top[state_samples[j,0]])[state_samples[j,1]])
            abc_t2_top.append(np.load(A_ring_T2_dist_top[state_samples[j,0]])[state_samples[j,1]])
            d_t1_bot.append(np.load(D_ring_T2_dist_bot[state_samples[j,0]])[state_samples[j,1]])
            d_t2_bot.append(np.load(D_ring_T2_dist_bot[state_samples[j,0]])[state_samples[j,1]])
        #print(i)
        #print(np.mean(abc_t1_top))
        #print(np.mean(abc_t2_top))
        #print(np.mean(d_t1_bot))
        #print(np.mean(d_t2_bot))

        if all((np.min(abc_t1_top) < 0.9, np.min(abc_t2_top) < 0.9, np.min(d_t1_bot) < 1.0, np.min(d_t2_bot) < 1.0)):
            anchored_states.append(i)
    
    return anchored_states

def id_inverse_anchored_states(samples):
    """Identify states where ligand is anchored to lid helices"""

    D_ring_T1_dist_top = sorted(glob.glob("../analysis/*D-ring*T1*top*"))
    D_ring_T2_dist_top = sorted(glob.glob("../analysis/*D-ring*T2*top*"))
    A_ring_T1_dist_bot = sorted(glob.glob("../analysis/*ABC*T1*bot*"))
    A_ring_T2_dist_bot = sorted(glob.glob("../analysis/*ABC*T2*bot*"))

    anchored_states = []

    for i in range(len(samples)):
        state_samples = samples[i]
        d_t1_top = []
        d_t2_top = []
        abc_t1_bot = []
        abc_t2_bot = []
        for j in range(len(state_samples)):
            d_t1_top.append(np.load(D_ring_T1_dist_top[state_samples[j,0]])[state_samples[j,1]])
            d_t2_top.append(np.load(D_ring_T2_dist_top[state_samples[j,0]])[state_samples[j,1]])
            abc_t1_bot.append(np.load(A_ring_T2_dist_bot[state_samples[j,0]])[state_samples[j,1]])
            abc_t2_bot.append(np.load(A_ring_T2_dist_bot[state_samples[j,0]])[state_samples[j,1]])

        if all((np.min(d_t1_top) < 0.9, np.min(d_t2_top) < 0.9, np.min(abc_t1_bot) < 1.0, np.min(abc_t2_bot) < 1.0)):
            anchored_states.append(i)
    
    return anchored_states

def id_loop_anchored_states(samples):
    """Identify states where ligand is anchored to D-loop"""

    A_ring_D_loop_dist_1 = sorted(glob.glob("../analysis/*ABC*D-loop*1*"))
    A_ring_D_loop_dist_3 = sorted(glob.glob("../analysis/*ABC*D-loop*3*"))
    D_ring_D_loop_dist_1 = sorted(glob.glob("../analysis/*D-ring*D-loop*1*"))
    D_ring_D_loop_dist_3 = sorted(glob.glob("../analysis/*D-ring*D-loop*3*"))

    anchored_states = []

    for i in range(len(samples)):
        state_samples = samples[i]
        abc_dloop_1 = []
        abc_dloop_3 = []
        d_ring_dloop_1 = []
        d_ring_dloop_3 = []
        for j in range(len(state_samples)):
            abc_dloop_1.append(np.load(A_ring_D_loop_dist_1[state_samples[j,0]])[state_samples[j,1]])
            abc_dloop_3.append(np.load(A_ring_D_loop_dist_3[state_samples[j,0]])[state_samples[j,1]])
            d_ring_dloop_1.append(np.load(D_ring_D_loop_dist_1[state_samples[j,0]])[state_samples[j,1]])
            d_ring_dloop_3.append(np.load(D_ring_D_loop_dist_3[state_samples[j,0]])[state_samples[j,1]])
        #print(i)
        #print(np.mean(abc_dloop_1))
        #print(np.mean(abc_dloop_3))
        #print(np.mean(d_ring_dloop_1))
        #print(np.mean(d_ring_dloop_3))

        if all((np.mean(abc_dloop_1) < 0.9, np.mean(abc_dloop_3) < 0.9, np.mean(d_ring_dloop_1) < 1.5, np.mean(d_ring_dloop_3) < 1.5)):
            anchored_states.append(i)
    
    return anchored_states

def coarse_grain(msm):

    pcca_object = msm.pcca(8)

    return pcca_object

def calc_tpt(msm, source, sink):

    tpt_object = pyemma.msm.tpt(msm, source, sink)
    mfpt = tpt_object.mfpt
    intermediates = tpt_object.I
    pathways = tpt_object.pathways

    return tpt_object, mfpt, intermediates, pathways

def pathway_fluxes(pathways, capacities, anchored):

    sticky_paths = []
    direct_paths = []
    sticky_flux = []
    direct_flux = []

    for i in range(len(pathways)):
        if any(j in anchored for j in pathways[i]):
            sticky_paths.append(pathways[i])
            sticky_flux.append(capacities[i])
        else:
            direct_paths.append(pathways[i])
            direct_flux.append(capacities[i])
    
    return sticky_paths, direct_paths, sticky_flux, direct_flux

def trim_state_sets(setA, setB):
    """Remove overlapping states between two sets of states"""

    overlap = np.intersect1d(setA, setB)
    setA_trimmed = [i for i in setA if i not in overlap]
    setB_trimmed = [i for i in setB if i not in overlap]

    return setA_trimmed, setB_trimmed, overlap

if __name__=="__main__":

    msm = pickle.load(open("msm_object.pkl",'rb'))
    #samples = get_samples(msm, topfile="../stripped.4ih4_withlig_noh.prmtop")
    #samples = get_samples(msm, topfile="../stripped.ShHTL7_withGR24.prmtop")
    #pickle.dump(samples, open("state_samples.pkl",'wb'))
    samples = pickle.load(open("tpt_old_bkp/state_samples.pkl",'rb'))
    bound, unbound, inverse = id_bound_unbound_states(samples)
    anchored = id_anchored_states(samples)
    inverse_anchored = id_inverse_anchored_states(samples)
    loop_anchored = id_loop_anchored_states(samples)

    ##Trimmed states
    anchored_trimmed, bound_trimmed, overlap = trim_state_sets(anchored, bound)
    print("ANCHORED TRIMMED")
    print(anchored_trimmed)
    print("BOUND TRIMMED")
    print(bound_trimmed)
    print("PARTIAL BOUND")
    print(overlap)
 
    unproductive_bound, _, _ = trim_state_sets(inverse, bound)
    print("UNPRODUCTIVE BINDING")
    print(unproductive_bound)

    unbound_trimmed, _, _ = trim_state_sets(unbound, anchored_trimmed)
    print("UNBOUND AND NOT ANCHORED")
    print(unbound_trimmed)

    print("Productive binding")
    tpt1, mfpt1, int1, path1 = calc_tpt(msm, unbound, bound_trimmed)
    print("MFPT:")
    print(mfpt1)
    print(tpt1.rate)

    print("Unbinding")
    tpt_unb, mfpt_unb, int_unb, path_unb = calc_tpt(msm, bound_trimmed, unbound)
    print("MFPT:")
    print(mfpt_unb)
    print(tpt_unb.rate)

    print("Anchoring")
    tpt_anc, mfpt_anc, int_anc, path_anc = calc_tpt(msm, unbound_trimmed, anchored_trimmed)
    print("MFPT:")
    print(mfpt_anc)
    print(tpt_anc.rate)

    print("Anchored to bound")
    tpt_atob, mfpt_atob, int_atob, path_anc = calc_tpt(msm, anchored_trimmed, bound_trimmed)
    print("MFPT:")
    print(mfpt_atob)
    print(tpt_atob.rate)

    ##Get network unbound to bound
    sets, tpt_cg = tpt1.coarse_grain([bound_trimmed, unbound_trimmed, np.concatenate((anchored_trimmed,overlap)), unproductive_bound])
    for s in sets:
        print(s)

    flux = tpt_cg.flux
    print(flux)

    print(10*flux/(np.max(flux)))

