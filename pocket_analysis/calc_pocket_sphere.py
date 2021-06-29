import numpy as np
import mdtraj as md

def get_slice(traj, resid_list):

    sel_string = "name CA and resid"
    for n in resid_list:
        sel_string += " " + str(n)

    slice_ind = traj.topology.select(sel_string)
    new_traj = traj.atom_slice(slice_ind)

    return new_traj

def compute_sphere(hel_traj, ser_traj):

    coords_hel = 10*hel_traj.xyz
    hel_center = np.mean(np.mean(coords_hel, axis=0), axis=0)
    
    coords_ser = 10*ser_traj.xyz[0,0,:]
    #R = np.linalg.norm(hel_center-coords_ser)

    midpoint = 0.5*(hel_center+coords_ser)
    R = np.linalg.norm(hel_center-midpoint)

    return coords_ser, R, midpoint

if __name__=="__main__":

    traj = md.load("4ih4_apo.pdb")
    resid_hel = range(132, 161)
    resid_ser = [92]

    #traj = md.load("ShHTL7_apo_minimized.pdb")
    #resid_hel = range(134, 163)
    #resid_ser = [94]

    hel_traj = get_slice(traj, resid_hel)
    ser_traj = get_slice(traj, resid_ser)

    coords_ser, R, midpoint = compute_sphere(hel_traj, ser_traj)
    print(coords_ser)
    print(R)
    print(midpoint)
