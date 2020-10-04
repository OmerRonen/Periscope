import logging
from argparse import ArgumentParser

from periscope.analysis.analyzer import save_model_predictions
from periscope.net.contact_map import get_model_by_name

logging.getLogger().setLevel(logging.CRITICAL)


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('protein', type=str, help='target name')
    parser.add_argument('-o', "--outfile", type=str, help='file to save prediction')

    # parser.add_argument('-p', '--proteins', nargs='+', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    protein = args.protein
    outfile = args.outfile
    model = get_model_by_name(model_name)
    save_model_predictions(model, protein, outfile)


if __name__ == '__main__':
    main()
