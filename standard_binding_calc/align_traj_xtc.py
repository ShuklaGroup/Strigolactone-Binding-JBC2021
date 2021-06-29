import numpy as np
import mdtraj as md
import glob
import re
import argparse
import os

def align_traj(ref, ref_atoms, trajfile, topfile, name):
    """Aligns trajectories based on given reference atoms"""

    #Ref should be mdtraj format
    
    traj = md.load(trajfile, top=topfile)
    traj_align = traj
    traj_align.superpose(ref, atom_indices=ref_atoms)

    if not os.path.isdir("./free_energy_calc/xtc_trajs"):
        os.mkdir("./free_energy_calc/xtc_trajs")

    traj_align.save_xtc("free_energy_calc/xtc_trajs/%s.xtc"%name)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, nargs=1, help="Directory containing trajectories in .nc format")
    parser.add_argument("--top_file", type=str, nargs=1, help="Topology file for trajectories")
    parser.add_argument("--ref_file", type=str, nargs=1, help="Reference structure file")
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()

    ref = md.load(args.ref_file)
    ref_top = ref.topology
    ref_atoms = ref_top.select("backbone")

    topfile = args.top_file

    for traj_file in glob.glob("%s/*"%args.traj_dir):
        name = re.sub(".nc", "", traj_file)
        name = re.sub("%s/"%args.traj_dir, "", name)
        align_traj(ref, ref_atoms, traj_file, topfile, name)
