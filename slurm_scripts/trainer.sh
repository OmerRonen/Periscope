#!/usr/bin/env bash

sbatch -J $1 --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il -o $1.out --gres gpu:1 --mem=48g --time=100:0:0 -c4 /vol/sci/bio/data/or.zuk/projects/ContactMaps/src/Periscope/slurm_scripts/train.sh $1 $2 $3 $4 $5 $6


