import tempfile, re
from pathlib import Path
from neuralmem.connectors.auto_discover import AutoDiscoveryEngine

with tempfile.TemporaryDirectory() as tmpdir:
    gitconfig = Path(tmpdir) / '.gitconfig'
    gitconfig.write_text('[user]\nname = test\n')
    
    # Test max_depth=0
    files0 = list(AutoDiscoveryEngine._walk_files(Path(tmpdir), 0))
    print('Files with depth=0:', [str(f) for f in files0])
    
    # Test max_depth=1
    files1 = list(AutoDiscoveryEngine._walk_files(Path(tmpdir), 1))
    print('Files with depth=1:', [str(f) for f in files1])
    
    engine = AutoDiscoveryEngine(scan_paths=[tmpdir], max_depth=1)
    results = engine.scan()
    print('Results with depth=1:', [(r.connector_name, r.confidence) for r in results])
