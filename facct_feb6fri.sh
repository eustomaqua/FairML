#!/bin/sh
# source activate py38|ensem


EXP=mCV_exp11h  # g
NC=11  # 17,13
DELT=.03
for ENS in Bagging AdaBoostM1 SAMME
do
    for DAT in ricci german ppr ppvr # adult
    do
        python -W ignore wp1_main_exec.py -add -exp $EXP --name-ens $ENS --nb-cls $NC --delta $DELT --eta .6 -dat $DAT -nk 5  # --logged
    done
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
