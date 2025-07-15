# FairML

TBA

<!--
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/eustomaqua/FairML/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/eustomaqua/FairML/tree/master) 
[![Coverage Status](https://coveralls.io/repos/github/eustomaqua/FairML/badge.svg?branch=master)](https://coveralls.io/github/eustomaqua/FairML?branch=master) 
https://readthedocs.org/projects/fairml/
-->
![CircleCI](https://img.shields.io/circleci/build/github/eustomaqua/FairML/master) 
[![Documentation Status](https://readthedocs.org/projects/fairml/badge/?version=latest)](https://fairml.readthedocs.io/en/latest/?badge=latest) 
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7676a4d116f447e897b6a4260b5c5f4f)](https://app.codacy.com/gh/eustomaqua/FairML/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) 
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/7676a4d116f447e897b6a4260b5c5f4f)](https://app.codacy.com/gh/eustomaqua/FairML/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage) 


### Getting started

https://fairml.readthedocs.io/en/latest/

```shell
$ rsync -r FairML hendrix:/home/qgl539/GitH/
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



<!--
### Acknowledgements

This research is funded by the European Union (MSCA, FairML, project no. 101106768). 

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
-->
