import os
import tempfile

import numpy as np

from argparse import ArgumentParser

from ..utils.constants import DATASETS, PATHS


def parse_args():
    parser = ArgumentParser(description="Calculate pssm and other scores")
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('-b', '--batch_size', type=int, help='dataset name', default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    targets = getattr(DATASETS, ds)
    bs = args.batch_size
    n_jobs = int(np.ceil(len(targets)/bs))

    for job in range(n_jobs-1):
        job_prots = targets[(job*bs):((job+1)*bs)]
        job_name = f'{ds}_{job}'
        with tempfile.NamedTemporaryFile(suffix='.sh') as f:
            with open(f.name, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --job-name=%s_tm\n" % job_name)
                fh.writelines(f"#SBATCH --output={os.path.join(PATHS.periscope, 'slurm_scripts', job_name + '.scoreout')}\n")
                fh.writelines("#SBATCH --time=24:0:0\n")
                fh.writelines("#SBATCH --mem=4g\n")
                fh.writelines("#SBATCH --mail-type=FAIL\n")
                fh.writelines("#SBATCH --mail-user=omer.ronen@mail.huji.ac.il\n")
                fh.writelines(f"python3 -m periscope.tools.calculate_scores -t {' '.join(job_prots)}")

            os.system("sbatch %s" % f.name)


if __name__ == '__main__':
    main()
