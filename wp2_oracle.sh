#!/bin/sh
# The partition is the queue you want to run on. standard is gpu and can be ommitted.
# number of independent tasks we are going to start in this script
# number of cpus we want to allocate for each program
# We expect that our program should not run longer than 2 days
# Note that a program will be killed once it exceeds this time!
# Skipping many options! see man sbatch


#SBATCH -p gpu --gres=gpu:0
#SBATCH --job-name=icml25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00


# From here on, we can start our program

# chmod +x wp2_oracle.sh
# source activate ensem


python wp1_main_exec.py --logged -exp mCV_expt3 --name-ens Bagging --abbr-cls DT --nb-cls 11 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt11 --name-ens Bagging --abbr-cls DT --nb-cls 11 --nb-pru 5 --delta 1e-6 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt8 --name-ens Bagging --abbr-cls DT --nb-cls 11 --nb-pru 5 -dat ricci
python wp1_main_exec.py --logged -exp mCV_expt10 --name-ens Bagging --abbr-cls DT --nb-cls 11 --nb-pru 5 --nb-lam 9 --nb-iter 2 -dat ricci
