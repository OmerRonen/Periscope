import os
import tempfile

from argparse import ArgumentParser

from ..utils.constants import DATASETS, PATHS


def parse_args():
    parser = ArgumentParser(description="Calculate model tm scores vs modeller and upload to Drive")
    parser.add_argument('-n', '--name', type=str, help='model name')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('-s', '--sswt', type=str, help='secondary structure weight', default='5')
    parser.add_argument('-r', '--selectrr', type=str, help='number of restraints', default='all')

    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    targets = getattr(DATASETS, ds)
    model_name = args.name
    r = args.selectrr
    s = args.sswt

    for target in targets:
        with tempfile.NamedTemporaryFile(suffix='.sh') as f:
            with open(f.name, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --job-name=%s_tm\n" % target)
                fh.writelines(f"#SBATCH --output={os.path.join(PATHS.periscope, 'slurm_scripts', target+'.tmout')}\n")
                fh.writelines("#SBATCH --time=100:0:0\n")
                fh.writelines("#SBATCH --mem=32g\n")
                fh.writelines("#SBATCH --mail-type=FAIL\n")
                fh.writelines("#SBATCH --mail-user=omer.ronen@mail.huji.ac.il\n")
                fh.writelines(f"python3 -m periscope.tools.calculate_model_tm -n {model_name} -t {target} -r {r} -s {s}")

            os.system("sbatch %s" % f.name)


if __name__ == '__main__':
    main()
