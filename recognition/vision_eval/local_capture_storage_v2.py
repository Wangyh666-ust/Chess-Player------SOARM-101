import hashlib
import io
import json
import os
import re
import shutil
import stat
import zipfile

__all__ = [
    "CaptureStorageError",
    "validate_capture_zip",
    "save_capture",
    "validate_root",
]

_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
_MAX_ZIP = 64 * 1024 * 1024
_MAX_TOTAL = 64 * 1024 * 1024
_MAX_JSON = 4 * 1024 * 1024
_MAX_JPEG = 24 * 1024 * 1024
_MAX_MEMBERS = 7
_MIN_MEMBERS = 3
_MIN_FREE = 2 * 1024 * 1024 * 1024
_CHUNK = 64 * 1024


class CaptureStorageError(ValueError):
    pass


def _fail(msg):
    raise CaptureStorageError(msg)


def _check_id(expected_id):
    if not isinstance(expected_id, str) or not _ID_RE.fullmatch(expected_id):
        _fail("invalid capture id")


def _member_name_ok(name):
    if not isinstance(name, str) or not name:
        return False
    if "\x00" in name:
        return False
    if name.startswith("/") or name.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:", name):
        return False
    if name.startswith("//") or name.startswith("\\\\"):
        return False
    parts = re.split(r"[/\\]", name)
    for part in parts:
        if part in ("", ".", ".."):
            return False
    return True


def _entry_type_ok(info):
    mode = info.external_attr >> 16
    ftype = stat.S_IFMT(mode)
    if ftype == 0:
        pass
    elif ftype == stat.S_IFREG:
        pass
    else:
        return False
    if info.is_dir():
        return False
    if info.flag_bits & 0x1:
        return False
    return True


def _read_member(zf, info, limit):
    try:
        with zf.open(info, "r") as fh:
            data = fh.read(limit + 1)
    except CaptureStorageError:
        raise
    except Exception as exc:
        raise CaptureStorageError("failed to read member: %s" % exc)
    if len(data) > limit:
        _fail("member too large: %s" % info.filename)
    return data


def _parse_cameras(pair, expected_id):
    cameras = pair.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 2:
        _fail("cameras must be a list of exactly two entries")
    by_index = {}
    numbers = set()
    for cam in cameras:
        if not isinstance(cam, dict):
            _fail("camera entry must be object")
        idx = cam.get("camera_index")
        num = cam.get("camera_number")
        if not isinstance(idx, int) or isinstance(idx, bool):
            _fail("camera_index invalid")
        if not isinstance(num, int) or isinstance(num, bool):
            _fail("camera_number invalid")
        if idx not in (0, 1):
            _fail("camera_index out of range")
        if idx in by_index:
            _fail("duplicate camera_index")
        if num not in (1, 2):
            _fail("camera_number out of range")
        if num in numbers:
            _fail("duplicate camera_number")
        if idx != num - 1:
            _fail("camera_index/number mismatch")
        by_index[idx] = cam
        numbers.add(num)
    if set(by_index) != {0, 1}:
        _fail("missing cameras")
    return [by_index[0], by_index[1]]


def _validate_camera_files(cam, expected_id, member_set):
    num = cam["camera_number"]
    prefix = "%s_camera%d_" % (expected_id, num)
    expected = {}
    for field, suffix in (
        ("original_file", "original.jpg"),
        ("upright_file", "upright.jpg"),
        ("yolo_file", "yolo.jpg"),
    ):
        val = cam.get(field)
        if field == "yolo_file" and val is None:
            continue
        if not isinstance(val, str) or not val:
            _fail("missing %s for camera %d" % (field, num))
        if not _member_name_ok(val):
            _fail("invalid filename for %s" % field)
        want = prefix + suffix
        if val != want:
            _fail("unexpected %s filename" % field)
        if val in expected:
            _fail("duplicate file reference")
        expected[val] = field
    src = cam.get("source_sha256")
    if not isinstance(src, str) or not re.match(r"^[0-9a-fA-F]{64}$", src):
        _fail("invalid source_sha256 for camera %d" % num)
    pred = cam.get("prediction_error", None)
    if pred is not None and not isinstance(pred, str):
        _fail("invalid prediction_error")
    grid = cam.get("grid", None)
    if grid is not None and not isinstance(grid, dict):
        _fail("invalid grid")
    return expected


def _check_image_suffix(name):
    return name.endswith(".jpg") or name.endswith(".jpeg")


def validate_capture_zip(blob, expected_id):
    if not isinstance(blob, (bytes, bytearray)):
        _fail("blob must be bytes")
    blob = bytes(blob)
    _check_id(expected_id)
    if len(blob) > _MAX_ZIP:
        _fail("zip exceeds size limit")
    if not blob:
        _fail("empty zip")

    try:
        zf = zipfile.ZipFile(io.BytesIO(blob), "r")
    except Exception as exc:
        raise CaptureStorageError("invalid zip: %s" % exc)

    with zf:
        infos = zf.infolist()
        if len(infos) < _MIN_MEMBERS or len(infos) > _MAX_MEMBERS:
            _fail("member count out of range")
        names = []
        total = 0
        seen = set()
        for info in infos:
            if not _member_name_ok(info.filename):
                _fail("invalid member name")
            if info.filename in seen:
                _fail("duplicate member: %s" % info.filename)
            seen.add(info.filename)
            if not _entry_type_ok(info):
                _fail("unsupported entry type: %s" % info.filename)
            total += info.file_size
            if total > _MAX_TOTAL:
                _fail("uncompressed total exceeds limit")
            names.append(info.filename)

        if "pair.json" not in seen:
            _fail("missing pair.json")

        pair_bytes = None
        pair_info = None
        for info in infos:
            if info.filename == "pair.json":
                pair_info = info
                break
        pair_bytes = _read_member(zf, pair_info, _MAX_JSON)
        if len(pair_bytes) == 0:
            _fail("empty pair.json")
        try:
            pair = json.loads(pair_bytes.decode("utf-8"))
        except CaptureStorageError:
            raise
        except Exception as exc:
            raise CaptureStorageError("invalid pair.json: %s" % exc)
        if not isinstance(pair, dict):
            _fail("pair.json must be object")
        if pair.get("id") != expected_id:
            _fail("pair id mismatch")

        cams = _parse_cameras(pair, expected_id)

        member_files = {}
        sha_by_name = {}
        expected_members = {"pair.json"}
        for cam in cams:
            file_map = _validate_camera_files(cam, expected_id, expected_members)
            for name, field in file_map.items():
                if not _check_image_suffix(name):
                    _fail("bad image extension: %s" % name)
                expected_members.add(name)
                member_files[name] = cam
                sha_by_name[name] = (cam, field)

        if set(names) != expected_members:
            extra = set(names) - expected_members
            missing = expected_members - set(names)
            if missing:
                _fail("missing member(s): %s" % ",".join(sorted(missing)))
            if extra:
                _fail("unexpected member(s): %s" % ",".join(sorted(extra)))

        files = {}
        files["%s_pair.json" % expected_id] = pair_bytes

        originals = {}
        for cam in cams:
            num = cam["camera_number"]
            name = cam["original_file"]
            originals[num] = name

        info_by_name = {info.filename: info for info in infos}
        for cam in cams:
            num = cam["camera_number"]
            orig_name = cam["original_file"]
            info = info_by_name[orig_name]
            if info.file_size > _MAX_JPEG:
                _fail("original image too large")
            data = _read_member(zf, info, _MAX_JPEG)
            digest = hashlib.sha256(data).hexdigest()
            if digest.lower() != cam["source_sha256"].lower():
                _fail("source_sha256 mismatch for camera %d" % num)
            files[orig_name] = data

        for name, (cam, field) in sha_by_name.items():
            if field == "original_file":
                continue
            info = info_by_name[name]
            if info.file_size > _MAX_JPEG:
                _fail("image too large: %s" % name)
            data = _read_member(zf, info, _MAX_JPEG)
            files[name] = data

    return {"id": expected_id, "pair": pair, "files": files}


def _has_control(s):
    for ch in s:
        o = ord(ch)
        if o < 0x20 or o == 0x7f:
            return True
    return False


def validate_root(root):
    if not isinstance(root, str) or not root:
        _fail("root must be non-empty string")
    if _has_control(root):
        _fail("root contains control characters")
    try:
        expanded = os.path.expanduser(root)
    except Exception:
        _fail("cannot expand root")
    if not expanded or not os.path.isabs(expanded):
        _fail("root must be absolute")
    if os.name == "nt":
        if expanded.startswith("\\\\") or expanded.startswith("//"):
            _fail("UNC path not supported")
        norm = expanded.replace("/", "\\")
        low = norm.lower()
        if low.startswith("\\\\.\\") or low.startswith("\\\\?\\"):
            _fail("device path not supported")
        import ntpath
        drive, tail = ntpath.splitdrive(norm)
        if not re.fullmatch(r"[A-Za-z]:", drive or ""):
            _fail("root must be absolute with drive letter")
        if not tail or tail[0] not in ("/", "\\"):
            _fail("root must be rooted on drive")
    else:
        if expanded.startswith("//"):
            pass
    try:
        canon = os.path.realpath(os.path.abspath(expanded))
    except Exception as exc:
        raise CaptureStorageError("cannot canonicalize root: %s" % exc)
    if not canon or not os.path.isabs(canon):
        _fail("canonical root invalid")
    if _has_control(canon):
        _fail("canonical root invalid")
    return canon


def _nearest_existing(path):
    cur = path
    while True:
        if os.path.exists(cur) or os.path.islink(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return cur
        cur = parent


def _free_space(path):
    try:
        usage = shutil.disk_usage(path)
    except Exception as exc:
        raise CaptureStorageError("cannot determine free space: %s" % exc)
    return usage.free


def _target_path(root, name):
    if not _member_name_ok(name) or os.sep in name or (os.altsep and os.altsep in name):
        _fail("unsafe target filename")
    if "/" in name or "\\" in name:
        _fail("unsafe target filename")
    target = os.path.join(root, name)
    real_root = os.path.realpath(root)
    real_target = os.path.realpath(target)
    if os.path.dirname(real_target) != real_root and real_root != os.path.dirname(real_target):
        if not (real_target == os.path.join(real_root, name)):
            _fail("target escapes root")
    if os.path.dirname(real_target) != real_root:
        _fail("target escapes root")
    return target


def save_capture(blob, root, expected_id):
    canon = validate_root(root)
    data = validate_capture_zip(blob, expected_id)

    if os.path.exists(canon):
        if not os.path.isdir(canon):
            _fail("root exists and is not a directory")
    else:
        if os.path.islink(canon):
            _fail("root is dangling or not a directory")

    files = data["files"]
    pair_base = "%s_pair.json" % expected_id
    ordered = sorted(files.keys())

    targets = []
    for name in ordered:
        target = _target_path(canon, name)
        if os.path.lexists(target):
            _fail("target already exists: %s" % name)
        targets.append((name, target))

    ancestor = _nearest_existing(canon)
    if not os.path.isdir(ancestor):
        ancestor = os.path.dirname(ancestor)
        if not os.path.isdir(ancestor):
            _fail("no existing ancestor directory")
    if _free_space(ancestor) < _MIN_FREE:
        _fail("insufficient free space")

    created = []
    created_dirs = []
    try:
        if not os.path.exists(canon):
            _mkdirs(canon, created_dirs)
        if not os.path.isdir(canon):
            _fail("root is not a directory")
        for name, target in targets:
            dest = os.path.join(canon, name)
            if os.path.lexists(dest):
                _fail("target already exists during write: %s" % name)
            try:
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
            except FileExistsError:
                raise CaptureStorageError("target appeared during write: %s" % name)
            except Exception as exc:
                raise CaptureStorageError("cannot create %s: %s" % (name, exc))
            created.append(dest)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(files[name])
                    fh.flush()
            except Exception as exc:
                raise CaptureStorageError("failed writing %s: %s" % (name, exc))
    except CaptureStorageError:
        _cleanup(created, created_dirs)
        raise
    except Exception as exc:
        _cleanup(created, created_dirs)
        raise CaptureStorageError("save failed: %s" % exc)

    return {
        "id": expected_id,
        "local_directory": canon,
        "pair_file": pair_base,
        "files": sorted(created),
    }


def _mkdirs(path, created_dirs):
    missing = []
    cur = path
    while not os.path.exists(cur) and not os.path.islink(cur):
        missing.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    for d in reversed(missing):
        try:
            os.mkdir(d)
            created_dirs.append(d)
        except FileExistsError:
            if not os.path.isdir(d):
                _fail("cannot create directory: %s" % d)
        except Exception as exc:
            raise CaptureStorageError("cannot create directory: %s" % exc)


def _cleanup(created, created_dirs):
    for path in reversed(created):
        try:
            if os.path.lexists(path):
                os.unlink(path)
        except Exception:
            pass
    for d in reversed(created_dirs):
        try:
            if os.path.isdir(d) and not os.path.islink(d):
                os.rmdir(d)
        except Exception:
            pass
