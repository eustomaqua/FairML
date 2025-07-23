# Set up the environment


We developed [FairML](https://github.com/eustomaqua/FairML) with `Python 3.8` and released the code to help you reproduce our work. Note that the experimental parts must be run on the `Ubuntu` operating system due to FairGBM (one baseline method that we used for comparison).


## Configuration

### Executing via Docker

```shell
$ # docker --version
$ # docker pull continuumio/miniconda3

$ cd ~/GitH*/FairML
$ # touch Dockerfile
$ # vim Dockerfile             # i  # Esc :wq
$ docker build -t fairgbm .    # <image-name>
$ docker images                # docker images -f dangling=true
$ docker run -it fairgbm /bin/bash


# cd home
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash ./Miniconda3-latest-Linux-x86_64.sh

Do you accept the license terms? [yes|no]
>>> yes
Miniconda3 will now be installed into this location:
[/root/miniconda3] >>> /home/miniconda3
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> no


# vim ~/.bashrc
export PATH=/home/miniconda3/bin:$PATH  # added by Anaconda3 installer

# source ~/.bashrc
# conda env list
# exit
$ docker ps -a                  # docker container ps|list
$ docker rm <container-id>
$ docker image rm <image-name>  # docker rmi <image-id>
```


### Executing on the server

```shell
$ ssh hendrix
$ srun -p gpu --pty --time=2-00:00:00 --gres gpu:0 bash
$ module load singularity
$ # mkdir Singdocker
$ cd Singdocker
$ singularity build --sandbox miniconda3 docker://continuumio/miniconda3

$ singularity shell --writable miniconda3   # <image-name>
Singularity> ls && conda env list
Singularity> conda create -n test python=3.8
Singularity> source activate test
(test) Singularity> pip list
(test) Singularity> conda deactivate
Singularity> exit

$ singularity build enfair.sif miniconda3/  # <environment-name>
$ # singularity instance list
$ # singularity cache list -v
$ # singularity cache clean
$ # singularity exec enfair.sif /bin/echo Hello World!
$ singularity shell enfair.sif              # singularity run *.sif

$ rm enfair.sif
$ yes | rm -r miniconda3
[qgl539@hendrixgpu04fl Singdocker]$ exit
[qgl539@hendrixgate03fl ~]$ exit
logout
```


## Requirements


### Python packages

```shell
$ # Install Anaconda/miniconda if you didn't
$ # To create a virtual environment
$ conda create -n test python=3.8
$ conda env list
$ source activate test
$
$ # To install packages
$ pip list && cd ~/FairML
$ pip install -U pip
$ pip install -r requirements.txt
$ python -m pytest
$
$ # To delete the virtual environment
$ conda deactivate && cd ..
$ yes | rm -r FairML
$ conda remove -n test --all
```


## Implementation

### Executing via Docker

```shell
$ docker ps -a
$ docker cp /home/yijun/<folder> <container-id>:/root/  # copy to docker

$ docker restart <container-id>
$ docker exec -it <container-id> /bin/bash
(base) # cd root/FairML
(base) # conda activate test
(test) # ....
(test) # conda deactivate
(base) # exit

$ docker cp <container-id>:/root/<folder> /home/yijun/  # copy from docker
$ docker stop <container-id>
```


### Executing on the server

```shell
$ rsync -r FairML hendrix:/home/qgl539/GitH/  # copy to server
$ ssh hendrix
$ screen                                      # screen -r <pts-id>
$ srun -p gpu --pty --time=23:30:00 --gres gpu:0 bash
$ module load singularity
$ cd Singdocker
$ singularity run enfair.sif

Singularity> cd ~/GitH/FairML
Singularity> source activate ensem
(ensem) Singularity> # executing ....
(ensem) Singularity> conda deactivate && cd ..
(base) Singularity> tar -czvf tmp.tar.gz FairML  # compression
(base) Singularity> yes | rm -r FairML

(base) Singularity> exit
[qgl539@hendrixgpu04fl Singdocker]$ exit
[qgl539@hendrixgate01fl ~]$ exit    # exit screen
[qgl539@hendrixgate01fl ~]$ logout  # Connection to hendrixgate closed.
$ rsync -r hendrix:/home/qgl539/tmp.tar.gz .  # copy from server
$ tar -xzvf tmp.tar.gz                        # decompression
$ rm tmp.tar.gz
```
