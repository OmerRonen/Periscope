#!/usr/bin/env bash
for i in {1..20}
do
   sbatch -J pfam_$i --mail-type=FAIL --mail-user=omer.ronen@mail.huji.ac.il -o pfam_$i.out --mem=32g --time=100:0:0 -c2 data_gen.sh $i 20 pfam
done
