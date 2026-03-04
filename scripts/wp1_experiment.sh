#!/bin/sh
# source activate py38|ensem


EXP=mCV_expt11
ENS=Bagging
DAT=ricci
for CLF in DT NB SVM linSVM LR1 LR2 LM1 LM2 kNNu kNNd MLP
do
    python wp1_main_exec.py -add --logged -exp $EXP --name-ens $ENS --abbr-cls $CLF --nb-cls 11 --delta 0.01 --eta .6 -dat $DAT # -nk 2
done
DAT=german
for CLF in DT NB SVM linSVM LR1 LR2 LM1 LM2 kNNu kNNd MLP
do
    python wp1_main_exec.py -add --logged -exp $EXP --name-ens $ENS --abbr-cls $CLF --nb-cls 11 --delta 0.01 --eta .6 -dat $DAT # -nk 2
done


# ssh hendrix
# srun -p gpu --pty --time=2-00:00:00 --gres gpu:0 bash
# module load singularity
# cd Singdocker
# singularity run enfair.sif
# exit

# chmod +x ?.sh
# nohup ./?.sh
# ps aux | grep FairML
# kill -9 *
# ps -ef | grep qgl539
