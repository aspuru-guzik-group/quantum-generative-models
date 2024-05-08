#!/bin/bash

# Input JSON file
# input_file="benchmark_models_settings_classical.json"

# for name in 
# do


for prior in "qcbm" "classical" "rbm"; do
# for i in 0 1 3 4 7 9; do
#   sbatch submit_docking.sh $i
        for size in $(seq 2 2 16); do
            # echo $cluster stoned_hill_climming_tartarus.py $protein qvina 4000 hill_climming/01_initial_stoned_qvina_$protein.csv
            # sbatch hill_climing_docking_cluster.sh $protein $cluster 250 $program hill_climming/initial_data.csv
            # echo "benchmark_models_submit_v0" $protein $size
            sbatch benchmark_models_submit_v0.sh $prior $size
    done
done