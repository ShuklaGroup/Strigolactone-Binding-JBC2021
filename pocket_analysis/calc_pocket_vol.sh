#!/bin/bash

while read line;
do

    n_frames=$(grep "MODEL" pdb_trajs\/${line}.pdb | wc -l)
    sed -i "18s/.*/NumFrames $n_frames/" AtD14_POVME2_input.ini
    sed -i "16s/.*/PDBFileName        pdb_trajs\/${line}.pdb/" AtD14_POVME2_input.ini
    sed -i "32s/.*/OutputFilenamePrefix        .\/POVME_output_manual_sphere\/${line}./" AtD14_POVME2_input.ini
    python -m POVME2 AtD14_POVME2_input.ini

    rm POVME_output_manual_sphere/*frame*.pdb
    rm POVME_output_manual_sphere/*output.txt

done < traj_list
