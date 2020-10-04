#!/usr/bin/env bash
for i in {1..76}
do
   sbatch -J cameo_$i --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il  -o cameo_$i.out --mem=32g --time=100:0:0 -c16 data_gen.sh $i 76 cameo
done
