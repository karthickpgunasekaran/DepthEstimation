#!/bin/bash
#
#SBATCH --job-name=par_original_withpose_modified
#SBATCH --output=out_par.txt     # output file
#SBATCH --partition=titanx-long
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00         # Runtime in D-HH:MM
#SBATCH --mem=10G
#SBATCH --mem-per-cpu=4096    # Memory in MB per cpu allocated

module load python/3.6.1
pip3 install --upgrade --user Pillow
python3 Main_parallel.py 
hostname
sleep 1
exit
