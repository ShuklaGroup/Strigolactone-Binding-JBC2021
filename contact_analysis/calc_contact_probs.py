"""Compute MSM-weighted residue-ligand contact probabilities"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import glob
import pickle

plt.rc('savefig', dpi=300)
matplotlib.rc('font',family='Helvetica-Normal',size=16)

def compute_contact_distances(traj_list, top, lig_resid):
    #Generate lig-protein residue pairs
    contacts_list = []
    for i in range(lig_resid):
        contacts_list.append([i, lig_resid])
    
    dist_all = []
    for traj_file in traj_list:
        traj = md.load(traj_file, top=top)
        dist_all.append(md.compute_contacts(traj, contacts=contacts_list, scheme='closest-heavy', ignore_nonprotein=False)[0])
    
    contact_distances = np.vstack(dist_all)

    return contact_distances

def compute_contact_probs(contact_distances, cutoff=0.45, weights=None):

    #Identify contacts
    contacts = contact_distances.copy() #Copy array
    print(contacts)
    contacts[contact_distances > cutoff] = 0.0 #Not in contact
    contacts[contact_distances < cutoff] = 1.0 #In contact

    print(contacts)
    #Compute ligand contact probability by residue
    if weights is None:
        contact_probs = np.sum(contacts, axis=0)/np.shape(contacts)[0]
    else:
        contact_probs = np.matmul(contacts.T, weights)
        #contact_probs = probs/np.sum(probs)

    return contact_probs

def compute_eq_contact_probs(contact_distances, dtrajs, msm, cutoff=0.45):

    contacts = contact_distances.copy()
    contacts[contact_distances > cutoff] = 0.0 #Not in contact
    contacts[contact_distances < cutoff] = 1.0 #In contact

    eq_dist = msm.pi
    dtrajs = np.hstack(dtrajs)

    contact_prob_per_state = np.zeros((np.shape(contacts)[1], np.shape(eq_dist)[0]))

    print(np.shape(contact_prob_per_state))

    for i in range(np.shape(contacts)[0]): #Count per-residue in each MSM state
        contact_prob_per_state[:,dtrajs[i]] += contacts[i,:].T

    for i in range(np.shape(eq_dist)[0]): #Normalize by MSM state count
        contact_prob_per_state[:,i] /= len(dtrajs[dtrajs==i])

    eq_contact_probs = np.matmul(contact_prob_per_state, eq_dist)

    return eq_contact_probs

def plot_contact_probs(contact_probs):

    plt.plot(contact_probs)
    plt.xlabel("Residue")
    plt.ylabel("Contact Probability")
    plt.tight_layout()
    plt.savefig("contact_prob.png")

if __name__=="__main__":

    contact_distances = compute_contact_distances(sorted(glob.glob("../stripped/*")), "../stripped.ShHTL7_withGR24.prmtop", 269)
    #contact_distances = compute_contact_distances(sorted(glob.glob("../stripped/*")), "../stripped.4ih4_withlig_noh.prmtop", 263)
    np.save("contact_distances_sorted.npy", contact_distances)

    contact_distances = np.load("contact_distances_sorted.npy")
    msm_weights = pickle.load(open("../msm_final_fixed/msm_weights.pkl",'rb'))

    contact_probs = compute_contact_probs(contact_distances, weights=np.concatenate(msm_weights), cutoff=0.4)

    np.save("contact_probs_weighted_4Acutoff.npy", contact_probs)

    msm = pickle.load(open("../msm_final_fixed/msm_object.pkl",'rb'))
    dtrajs = pickle.load(open("../msm_final_fixed/dtrajs.pkl",'rb'))

    eq_contact_probs = compute_eq_contact_probs(contact_distances, dtrajs, msm, cutoff=0.4)
    np.save("eq_contact_probs_final.npy", eq_contact_probs)

    contact_probs = np.load("contact_probs_weighted_4Acutoff.npy")
    plot_contact_probs(contact_probs)
