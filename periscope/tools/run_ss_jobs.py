import os
import tempfile

from argparse import ArgumentParser

from ..utils.constants import DATASETS, PATHS


def parse_args():
    parser = ArgumentParser(description="run raptor x to predicts ss")
    parser.add_argument('datasets', nargs='+', default=[], help='datasets')
    parser.add_argument('n', type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    n = args.n
    for d in args.datasets:
        proteins = getattr(DATASETS, d)
        if n > 0:
            proteins = proteins[0:n]
        for protein in proteins:
            # with tempfile.NamedTemporaryFile(suffix='.sh') as f:
            f = os.path.join(PATHS.periscope, 'slurm_scripts', "job_"+protein + '.sh')
            # with open(f.name, 'w') as fh:
            if True:
                with (open(f, 'w')) as fh:
                    fh.writelines("#!/usr/bin/env bash\n")
                    fh.writelines("#SBATCH --job-name=%s_tm\n" % protein)
                    fh.writelines(
                        f"#SBATCH --output={os.path.join(PATHS.periscope, 'slurm_scripts', protein + '.raptor_ss')}\n")
                    fh.writelines("#SBATCH --time=100:0:0\n")
                    fh.writelines("#SBATCH --mem=32g\n")
                    fh.writelines("#SBATCH -M hm\n")
                    fh.writelines("#SBATCH --mail-type=FAIL\n")
                    fh.writelines("#SBATCH --mail-user=omer.ronen@mail.huji.ac.il\n")
                    fh.writelines(
                        f"python3 -m periscope.tools.ss_raptor {protein}")

                # os.system("sbatch %s" % f.name)
                os.system("sbatch %s" % f)


if __name__ == '__main__':
    main()
