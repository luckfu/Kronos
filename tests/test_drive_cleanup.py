from finetune.drive_cleanup import cleanup_drive_conflict_files


class FakeRequest:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def execute(self):
        return self.payload


class FakeFiles:
    def __init__(self):
        self.deleted = []

    def list(self, **kwargs):
        assert kwargs['q'] == "'root' in parents"
        return FakeRequest({
            'files': [
                {'id': 'a', 'name': 'last_state.pt', 'trashed': False},
                {'id': 'b', 'name': 'last_state (12).pt', 'trashed': True},
                {'id': 'c', 'name': 'model (3).safetensors', 'trashed': False},
                {'id': 'd', 'name': 'other.pt', 'trashed': False},
                {'id': 'e', 'name': 'candidate.safetensors', 'trashed': False},
            ],
        })

    def delete(self, *, fileId):
        self.deleted.append(fileId)
        return FakeRequest()


class FakeService:
    def __init__(self):
        self.file_resource = FakeFiles()

    def files(self):
        return self.file_resource


def test_api_cleanup_permanently_deletes_only_checkpoint_conflicts(monkeypatch):
    monkeypatch.setenv('KRONOS_DRIVE_CONFLICT_CLEANUP', 'api')
    service = FakeService()

    removed = cleanup_drive_conflict_files(required=True, service=service)

    assert removed == [
        'last_state.pt',
        'last_state (12).pt',
        'model (3).safetensors',
    ]
    assert service.file_resource.deleted == ['a', 'b', 'c']


def test_api_cleanup_is_disabled_without_explicit_mode(monkeypatch):
    monkeypatch.delenv('KRONOS_DRIVE_CONFLICT_CLEANUP', raising=False)
    service = FakeService()

    assert cleanup_drive_conflict_files(required=True, service=service) == []
    assert service.file_resource.deleted == []
