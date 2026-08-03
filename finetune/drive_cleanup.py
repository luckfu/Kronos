"""Permanently remove Colab Drive root checkpoint conflict copies."""

import os
import re


DRIVE_CONFLICT_NAME = re.compile(
    r'^(?:last_state(?: \(\d+\))?\.pt|model(?: \(\d+\))?\.safetensors)$'
)
DRIVE_SCOPE = 'https://www.googleapis.com/auth/drive'
_DRIVE_SERVICE = None
_WARNING_SHOWN = False


def drive_cleanup_enabled():
    return os.getenv('KRONOS_DRIVE_CONFLICT_CLEANUP', '').strip().lower() == 'api'


def _get_drive_service():
    global _DRIVE_SERVICE
    if _DRIVE_SERVICE is None:
        import google.auth
        from googleapiclient.discovery import build

        credentials, _ = google.auth.default(scopes=[DRIVE_SCOPE])
        _DRIVE_SERVICE = build(
            'drive', 'v3', credentials=credentials, cache_discovery=False
        )
    return _DRIVE_SERVICE


def cleanup_drive_conflict_files(required=False, service=None):
    """Permanently delete matching files whose parent is the authenticated Drive root."""
    global _WARNING_SHOWN
    if not drive_cleanup_enabled():
        return []

    try:
        service = service or _get_drive_service()
        removed = []
        page_token = None
        while True:
            response = service.files().list(
                q="'root' in parents",
                spaces='drive',
                fields='nextPageToken, files(id, name, trashed)',
                pageSize=1000,
                pageToken=page_token,
            ).execute()
            for item in response.get('files', []):
                if not DRIVE_CONFLICT_NAME.fullmatch(item.get('name', '')):
                    continue
                service.files().delete(fileId=item['id']).execute()
                removed.append(item['name'])
            page_token = response.get('nextPageToken')
            if not page_token:
                break

        if removed:
            print(
                f"Permanently deleted {len(removed)} Drive root "
                "checkpoint conflict file(s)."
            )
        return removed
    except Exception as exc:
        if required:
            raise RuntimeError(
                "Google Drive API cleanup is required but unavailable. "
                "Run google.colab.auth.authenticate_user() in a notebook cell."
            ) from exc
        if not _WARNING_SHOWN:
            print(f"Warning: permanent Drive conflict cleanup failed: {exc}")
            _WARNING_SHOWN = True
        return []


if __name__ == '__main__':
    cleanup_drive_conflict_files(required=True)
    print('Google Drive permanent checkpoint cleanup is ready.')
