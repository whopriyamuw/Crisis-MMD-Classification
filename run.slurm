#!/bin/bash
#SBATCH --job-name=trans
#SBATCH --output=trans_%j.out
#SBATCH --error=trans_%j.err

#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

#SBATCH --export=all
#SBATCH --chdir=/gscratch/scrubbed/jcols/crisis-mmd/src
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jcols@uw.edu

eval "$(conda shell.bash hook)"
conda activate crisis

python -m models.translation