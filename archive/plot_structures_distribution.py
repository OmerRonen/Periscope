import os

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from .analysis import plot_structures_distribution, investigate_structures_distribution
from .drive import upload_folder
from .globals import DATASETS, periscope_path


def parse_args():
    parser = ArgumentParser(description="creates structures dist plot")
    parser.add_argument('-d', '--dataset', type=str, help='data set name')
    parser.add_argument('-s',
                        '--sample',
                        type=int,
                        help='number of targets to sample',
                        default=None)
    parser.add_argument('-k',
                        type=int,
                        help='number of references to consider',
                        default=None)
    parser.add_argument('-o',
                        '--outfolder',
                        type=str,
                        help='folder to save figure in')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    sample = args.sample
    k = args.k
    outfolder = args.outfolder
    if not os.path.exists(os.path.join(periscope_path, outfolder)):
        os.mkdir(outfolder)

    targets = DATASETS[dataset]
    if sample:
        targets = np.random.choice(targets, sample)

    out_dict, target_imprv = investigate_structures_distribution(
        targets, False, False, True, k)

    target_imprv_df = pd.DataFrame(target_imprv)

    target_imprv_df.to_csv(
        os.path.join(periscope_path, outfolder, 'potential.csv'))

    outfile = os.path.join(periscope_path, outfolder,
                           'structures_dist_%s_mistakes.png' % dataset)

    plot_structures_distribution(out_dict, outfile, nas=False)

    out_dict = investigate_structures_distribution(targets,
                                                   all_contacts=True,
                                                   nas=False,
                                                   k=k)

    outfile = os.path.join(periscope_path, outfolder,
                           'structures_dist_%s_full.png' % dataset)

    plot_structures_distribution(out_dict, outfile, nas=False)

    out_dict = investigate_structures_distribution(targets,
                                                   all_contacts=False,
                                                   nas=True,
                                                   k=k)

    outfile = os.path.join(periscope_path, outfolder,
                           'structures_dist_%s_nas.png' % dataset)

    plot_structures_distribution(out_dict, outfile, nas=True)

    out_dict = investigate_structures_distribution(targets,
                                                   all_contacts=False,
                                                   nas=True,
                                                   target_contact=False,
                                                   k=k)

    outfile = os.path.join(periscope_path, outfolder,
                           'structures_dist_%s_nas_no_contact.png' % dataset)

    plot_structures_distribution(out_dict, outfile, nas=True)

    upload_folder(os.path.join(periscope_path, outfolder), outfolder)


if __name__ == '__main__':
    main()
