#!/usr/bin/env bash
export PATH="/cs/cbio/orzuk/projects/ContactMaps/src/hh/bin:$PATH"
export PATH="/cs/zbio/orzuk/projects/ContactMaps/src/MSA_Completion/mmseqs2/bin:$PATH"
export PATH="/cs/zbio/orzuk/projects/ContactMaps/src/MSA_Completion/hashdeep/bin:$PATH"
export PATH="/cs/cbio/orzuk/projects/ContactMaps/src/plmc/bin:$PATH"
export PATH="/cs/cbio/orzuk/projects/ContactMaps/src/hh/scripts:$PATH"
export HHLIB="/cs/cbio/orzuk/projects/ContactMaps/src/hh"
module load tensorflow
python -W ignore /cs/zbio/orzuk/projects/ContactMaps/src/Periscope/Evaluate.py