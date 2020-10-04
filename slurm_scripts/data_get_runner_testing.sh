#!/usr/bin/env bash

sbatch -J testing --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il --gres gpu:1 -o testing_$i.out --mem=32g --time=100:0:0 -c2 data_gen.sh 1 1 testing
