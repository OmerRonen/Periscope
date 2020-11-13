import os
import logging

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
import googleapiclient.errors

from os import listdir
from sys import exit
import ast

from ..utils.constants import PATHS

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = os.path.join(
    PATHS.periscope, 'client_secrets.json')

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(os.path.join(PATHS.periscope, "mycreds.txt"))
    if gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(os.path.join(PATHS.periscope, "mycreds.txt"))

    gauth.Authorize()

    return GoogleDrive(gauth)


def _get_folder_id(drive, parent_folder_id, folder_name):
    file_list = GoogleDriveFileList()
    try:
        file_list = drive.ListFile({
            'q':
            "'{0}' in parents and trashed=false".format(parent_folder_id)
        }).GetList()
    # Exit if the parent folder doesn't exist
    except googleapiclient.errors.HttpError as err:
        # Parse error message
        message = ast.literal_eval(err.content)['error']['message']
        if message == 'File not found: ':
            print(message + folder_name)
            exit(1)
        # Exit with stacktrace in case of other error
        else:
            raise

    # Find the the destination folder in the parent folder's files
    for file1 in file_list:
        if file1['title'] == folder_name:
            return file1['id']


def _create_folder(drive, folder_name, parent_folder_id):
    folder_metadata = {
        'title': folder_name,
        # Define the file type as folder
        'mimeType': 'application/vnd.google-apps.folder',
        # ID of the parent folder
        'parents': [{
            "kind": "drive#fileLink",
            "id": parent_folder_id
        }]
    }

    folder = drive.CreateFile(folder_metadata)
    folder.Upload()

    return folder['id']


def _recursive_upload(drive, folder_id, src_name, exclude=[]):
    if src_name.split('/')[-1].startswith('tmp'):
        return
    if os.path.isfile(src_name):
        LOGGER.info('Uploading %s' % src_name)
        f_name = src_name.split('/')[-1]

        f = drive.CreateFile({
            "parents": [{
                "kind": "drive#fileLink",
                "id": folder_id
            }],
            'title':
            f_name
        })

        f.SetContentFile(src_name)
        f.Upload()
        return

    for file1 in listdir(src_name):
        LOGGER.info(f'Uploading {os.path.join(src_name, file1)}')
        if file1 in exclude:
            continue

        full_path = os.path.join(src_name, file1)
        is_folder = not os.path.isfile(full_path)
        sub_folder_id = folder_id

        if is_folder:
            sub_folder_id = _get_folder_id(drive, folder_id, file1)

            if not sub_folder_id:
                sub_folder_id = _create_folder(drive, file1, folder_id)
        _recursive_upload(drive, sub_folder_id, full_path, exclude=exclude)


def upload_folder(src_name, dst_name):
    drive = _get_drive()
    directory_id = _get_folder_id(drive, 'root', 'Periscope')
    for f in dst_name.split('/'):
        current_directory_id = _get_folder_id(drive, directory_id, f)
        if not current_directory_id:
            current_directory_id = _create_folder(drive, f, directory_id)
        directory_id = current_directory_id

    _recursive_upload(drive, directory_id, src_name)




