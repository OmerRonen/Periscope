#!/usr/bin/env bash
module load tensorflow
python3 -W ignore -m periscope.tools.trainer -n $1 $2 $3 $4 $5 $6