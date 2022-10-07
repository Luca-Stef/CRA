#!/bin/sh

#SBATCH --output ovito.out 
#SBATCH --error ovito.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

python gen_WS.py $1 $2 
#python gen_DXA.py $1 $2 
#python gen_C15.py $1 $2 
