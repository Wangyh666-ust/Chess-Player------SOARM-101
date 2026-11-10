from pathlib import Path

SOURCE_SHA = "cce7cde8cc1b4b2fe7b06767727b1a3e7c9ff2dd9ad4c34cf573eee2204750db"
_OLD_IMPORT = "from pilot_world import PilotWorld"
_NEW_IMPORT = "from pilot_world_v2 import PilotWorld"

_source_path = Path(__file__).with_name("query_world_reuse_v1.py")
_source = _source_path.read_text(encoding="utf-8")

import hashlib as _hashlib
_actual_sha = _hashlib.sha256(_source.encode("utf-8")).hexdigest()
assert _actual_sha == SOURCE_SHA, (
    f"query_world_reuse_v1.py SHA mismatch: {_actual_sha} != {SOURCE_SHA}"
)

_replacement_count = _source.count(_OLD_IMPORT)
assert _replacement_count == 1, (
    f"expected exactly 1 occurrence of {_OLD_IMPORT!r}, found {_replacement_count}"
)

_patched_source = _source.replace(_OLD_IMPORT, _NEW_IMPORT)

_code = compile(_patched_source, str(_source_path), "exec")
exec(_code, globals())
