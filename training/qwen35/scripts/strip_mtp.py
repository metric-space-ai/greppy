#!/usr/bin/env python3
"""Strip the NextN/MTP block (blk.<N>.*) from a converted Qwen3.5 GGUF and set
block_count back so greppy's qwen35-native loader accepts it.

Mirrors the KV/tensor copy pattern of llama.cpp's gguf_new_metadata.py.
Usage: strip_mtp.py SRC.gguf DST.gguf [drop_block=24]
"""
import sys
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
drop_block = int(sys.argv[3]) if len(sys.argv) > 3 else 24
drop_prefix = f"blk.{drop_block}."

r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
block_count_key = f"{arch}.block_count"
w = GGUFWriter(dst, arch)

for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == block_count_key:
        print(f"{field.name}: {val} -> {drop_block}")
        val = drop_block
    if field.name == f"{arch}.nextn_predict_layers" and val:
        print(f"{field.name}: {val} -> 0")
        val = 0
    w.add_key_value(field.name, val, val_type, sub_type=sub_type)

kept, dropped = [], 0
for t in r.tensors:
    if t.name.startswith(drop_prefix):
        dropped += 1
        continue
    kept.append(t)
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in kept:
    w.write_tensor_data(t.data)
w.close()
print(f"dropped {dropped} tensors under {drop_prefix}, kept {len(kept)}, wrote {dst}")
