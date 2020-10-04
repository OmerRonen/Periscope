#!/usr/bin/env bash
for i in {1..200}
do
   sbatch -J train_$i --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il -o train_$i.out --mem=32g --time=100:0:0 -c16 data_gen.sh $i 200 train
done
