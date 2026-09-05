import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from catalog_archive import catalog
from contracts import strict_json


class ParserInventoryTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_values_are_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":{"x":1,"x":2}}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.assertRaises(ValueError):
                strict_json(raw)
        self.assertEqual(strict_json('{"annotations":[]}'),{'annotations':[]})

    def test_inventory_does_not_admit_or_copy_private_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive=root/'raw.jsonl'
            raw=(json.dumps({'wall':'error 123\n','source':'repo/test','dataset':'public',
                             'exit_code':1,'next_turn':'PRIVATE_REASONING_MUST_NOT_BE_EXPORTED'})+'\n').encode()
            archive.write_bytes(raw)
            result=catalog(archive,root/'out')
            self.assertEqual(result['archive_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['counts']['records'],1)
            self.assertEqual(result['capture_verified'],0)
            self.assertEqual(result['eligible_final_sources'],0)
            self.assertNotIn(b'PRIVATE_REASONING', (root/'out/catalog.sqlite').read_bytes())
            with self.assertRaises(FileExistsError): catalog(archive,root/'out')


if __name__=='__main__': unittest.main()
