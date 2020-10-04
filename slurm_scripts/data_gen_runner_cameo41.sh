#!/usr/bin/env bash
for i in {1..41}
do
   sbatch -J cameo41_$i --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il -o cameo41_$i.out --mem=32g --time=100:0:0 -c16 data_gen.sh $i 41 cameo41
done
