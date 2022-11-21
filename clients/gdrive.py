from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


CONNECTOR = None


def connect():
    global CONNECTOR
    if not CONNECTOR:
        CONNECTOR = authenticate()
    return CONNECTOR


def authenticate():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def upload_file(path, parent_folder_id):
    """Upload file to Google Drive. Triggers authentication in the 1st time."""
    drive = connect()
    filename = path.split("/")[-1]
    gfile = drive.CreateFile(
        metadata={
            "parents": [{"id": parent_folder_id}],
            "title": filename
        }
    )
    gfile.SetContentFile(path)
    gfile.Upload()
