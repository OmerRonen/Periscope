#!/usr/bin/env bash

sbatch -J $1 --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il -o $1.out --mem=200g --time=100:0:0 -c8 /cs/zbio/orzuk/projects/ContactMaps/src/Periscope/slurm_scripts/python.sh -m periscope.tools.ensemble_prediction -n $1 -m ccmpred_ms_2 ccmpred_ms_3 ccmpred_ms_4 ccmpred_ms_5