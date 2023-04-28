# cse6230-spring23-final-proj-ddppo
Final project for cse6230 HPPC, focusing on decentralized distributed PPO: https://github.com/agaction/cse6230-spring23-final-proj-ddppo

`qsub -l walltime=02:00:00 -l nodes=1:ppn=4:gpus=1:teslav100 -l pmem=8gb -q coc-ice-gpu -I`

`git clone https://github.com/agaction/cse6230-spring23-final-proj-ddppo.git`

## Prepare the environment
`module load cuda/10.0 gcc/7.4.0 anaconda3/2019.07`

## Install CULE
`cd cse6230-spring23-final-proj-ddppo`

`./install_cule.sh`

## To run PPO
`python src/ppo_main.py -c configs/cpu_benchmark.config`

You can also vary the different parameters by modifying the benchmark.config or using the arguments:
```
--num-ales
--batch-size
--num-steps
```

There are 3 preconfigured options, cpu_benchmark.config will run a training session using only cpu, gpu_single.config will run on a single gpu and gpu_parallel.config attemps to run using all the available gpus but fails due to an issue with running distributed code on old versions of pytorch when the gpus are in exclusive process mode.



## To visualize a trained model playing pong
Train the model and save it to a .pth file:
`python src/ppo_main.py -c configs/benchmark.config --save-model-filename results/model_out.pth`

Make sure that you have the ability to see the gui popup, either using X11 forwarding from the gpu enabled node or by using an [OnDemand](https://docs.pace.gatech.edu/ood/guide/) interactive desktop session then run:
`python src/animate.py --env-name PongNoFrameskip-v4 --usecuda --model-filename results/model_out.pth`
