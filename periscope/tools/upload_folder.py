from argparse import ArgumentParser

from ..utils.drive import upload_folder


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('-s', '--source', type=str, help='Folder to upload')
    parser.add_argument('-d',
                        '--destination',
                        type=str,
                        help='Destination Folder in Google Drive')
    parser.add_argument('-e', '--exclude', nargs='+', default=[],
                        help='Folders to exclude')

    return parser.parse_args()


def main():
    args = parse_args()
    src_name = args.source
    dst_name = args.destination
    exclude = args.exclude

    upload_folder(src_name, dst_name, exclude)


if __name__ == "__main__":
    main()
