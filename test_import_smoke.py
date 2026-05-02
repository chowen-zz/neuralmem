#!/usr/bin/env python3
"""Smoke test for import_memories."""
import json
import sys
import tempfile
import os

sys.path.insert(0, 'src')
from neuralmem.core.memory import NeuralMem
from neuralmem.core.config import NeuralMemConfig

class MockEmbedder:
    dimension = 4
    def encode(self, texts):
        import hashlib
        results = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            v = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]
            norm = sum(x ** 2 for x in v) ** 0.5 or 1.0
            results.append([x / norm for x in v])
        return results
    def encode_one(self, text):
        return self.encode([text])[0]

tmp = tempfile.mkdtemp()
db = os.path.join(tmp, 'test.db')
cfg = NeuralMemConfig(db_path=db, embedding_dim=4)
embedder = MockEmbedder()
mem = NeuralMem(config=cfg, embedder=embedder)

# Store some memories
mem.remember('User prefers TypeScript')
mem.remember('User likes Python programming')

# Export as JSON
json_data = mem.export_memories(format='json')
print('Exported JSON:')
print(json_data[:200])

# Create new mem instance to test import
mem2_db = os.path.join(tmp, 'test2.db')
cfg2 = NeuralMemConfig(db_path=mem2_db, embedding_dim=4)
mem2 = NeuralMem(config=cfg2, embedder=embedder)

# Import from JSON
count = mem2.import_memories(json_data, format='json')
print(f'\nImported {count} memories from JSON')

# Verify
memories = mem2.storage.list_memories()
print(f'Total memories in mem2: {len(memories)}')
for m in memories:
    print(f'  - {m.content}')

# Test duplicate skip
count2 = mem2.import_memories(json_data, format='json', skip_duplicates=True)
print(f'\nRe-import (skip_duplicates=True): {count2} imported (should be 0)')

# Test markdown export/import
md_data = mem.export_memories(format='markdown')
print('\nMarkdown export:')
print(md_data[:300])

mem3_db = os.path.join(tmp, 'test3.db')
cfg3 = NeuralMemConfig(db_path=mem3_db, embedding_dim=4)
mem3 = NeuralMem(config=cfg3, embedder=embedder)
count3 = mem3.import_memories(md_data, format='markdown')
print(f'\nImported {count3} from markdown')

# Test CSV export/import
csv_data = mem.export_memories(format='csv')
print('\nCSV export:')
print(csv_data[:200])

mem4_db = os.path.join(tmp, 'test4.db')
cfg4 = NeuralMemConfig(db_path=mem4_db, embedding_dim=4)
mem4 = NeuralMem(config=cfg4, embedder=embedder)
count4 = mem4.import_memories(csv_data, format='csv')
print(f'\nImported {count4} from CSV')

# Test user_id override
mem5_db = os.path.join(tmp, 'test5.db')
cfg5 = NeuralMemConfig(db_path=mem5_db, embedding_dim=4)
mem5 = NeuralMem(config=cfg5, embedder=embedder)
count5 = mem5.import_memories(json_data, format='json', user_id='overridden-user')
memories5 = mem5.storage.list_memories()
print(f'\nWith user_id override: {count5} imported')
for m in memories5:
    print(f'  - user_id={m.user_id}: {m.content}')

# Test unsupported format
try:
    mem5.import_memories('data', format='xml')
except Exception as e:
    print(f'\nUnsupported format error: {e}')

print('\nAll smoke tests passed!')
