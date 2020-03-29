#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -p g
#SBATCH --gres=gpu:P100:1
#SBATCH --qos=medium
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

bash eval.sh ara_anthoSet