#!/usr/bin/env bash
for i in {1..50}
do
   sbatch -J eval_$i --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il  -o eval_$i.out --mem=32g --time=100:0:0 -c8 data_gen.sh $i 50 eval
done
