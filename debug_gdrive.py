import sys
sys.path.insert(0, '/Users/Zhuanz/Desktop/ai-agent-research/neuralmem-main/src')

from unittest.mock import MagicMock, patch
from neuralmem.connectors import gdrive as gdrive_module
from neuralmem.connectors.base import ConnectorState

with patch.object(gdrive_module, '_HAS_GOOGLE_API', True):
    connector = gdrive_module.GoogleDriveConnector(
        name='gdrive',
        config={
            'credentials': {'token': 't', 'refresh_token': 'r', 'client_id': 'c', 'client_secret': 's'},
            'folder_id': 'folder123',
            'mime_types': ['text/plain'],
        },
    )

mock_service = MagicMock()
files_mock = MagicMock()
mock_service.files.return_value = files_mock

files_result = {'files': [{'id': 'file1', 'name': 'doc1.txt', 'mimeType': 'text/plain', 'size': '12',
    'modifiedTime': '2024-01-01T00:00:00Z', 'createdTime': '2024-01-01T00:00:00Z',
    'owners': [{'displayName': 'Alice'}], 'webViewLink': 'https://drive.google.com/file/d/file1/view'}]}

mock_service.files.return_value.list.return_value.execute.return_value = files_result
mock_service.files.return_value.get_media.return_value = MagicMock()

with patch.object(gdrive_module, 'Credentials') as MockCreds, \
     patch.object(gdrive_module, 'build', return_value=mock_service), \
     patch.object(gdrive_module, 'MediaIoBaseDownload') as MockDownloader, \
     patch.object(gdrive_module, 'io') as MockIO:
    creds_instance = MagicMock()
    creds_instance.expired = False
    MockCreds.return_value = creds_instance
    
    connector.authenticate()
    
    fh_mock = MagicMock()
    fh_mock.getvalue.return_value = b'hello world'
    MockIO.BytesIO.return_value = fh_mock
    
    downloader_instance = MockDownloader.return_value
    downloader_instance.next_chunk.side_effect = [
        (MagicMock(progress=MagicMock(return_value=1.0)), True),
    ]
    
    print('state before sync:', connector.state)
    print('_service before sync:', connector._service)
    print('_available before sync:', connector._available)
    
    # Call sync and trace
    try:
        items = connector.sync(limit=10)
        print('sync items:', len(items))
    except Exception as e:
        print('sync exception:', e)
        import traceback
        traceback.print_exc()
    
    print('state after sync:', connector.state)
    print('last_error after sync:', connector.last_error())
