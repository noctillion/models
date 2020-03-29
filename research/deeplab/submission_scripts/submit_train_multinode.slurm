#!/bin/bash
#SBATCH --time=1-00:00:00 --qos=medium -N2 -n2 -c2 --mem-per-cpu=32G
#SBATCH packjob
#SBATCH --time=1-00:00:00 --qos=medium -N4 -n4 -c2 --mem=169G -p g --gres=gpu:V100:4

srun --mpi=none --pack-group=0,1 --ntasks-per-node=1 train_multinode.sh ara_anthoSet
