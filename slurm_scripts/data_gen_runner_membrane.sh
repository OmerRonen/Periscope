#!/usr/bin/env bash
for i in {1..398}
do
   sbatch -J membrane_$i --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il -o membrane_$i.out --mem=32g --time=100:0:0 -c4 data_gen.sh $i 398 membrane
done
