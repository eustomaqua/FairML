# Getting started





```shell
rsync -r FairML hendrix:/home/qgl539/GitH/
```

```shell
ssh hendrix
srun -p gpu --pty --time=23:30:00 --gres gpu:0 bash
screen
module load singularity
cd Singdocker
singularity run enfair.sif
cd GitH/FairML
source activate ensem

# tar -czvf FairML.tar.gz FairML
yes | rm -r FairML
```

