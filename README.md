# cse6230-spring23-final-proj-ddppo
Final project for cse6230 HPPC, focusing on decentralized distributed PPO


## 0. Request Node and GPU
`qsub -l walltime=02:00:00 -l nodes=1:ppn=4:gpus=1:teslav100 -l pmem=8gb -q coc-ice-gpu -I`

`module load anaconda3/2021.05`

`module load cuda/11.1`
## 1. INSTALL
(recomend create a conda environment first with python=3.9)
`pip3 install -r requirements.txt`

## 2. RUN

`python3 main.py --algo dppo`

