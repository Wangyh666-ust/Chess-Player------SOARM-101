import ipaddress
import json
import math
import os
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "vision_eval" / "dual_camera_launcher_config.json"
RESULTS_DIR = ROOT / "vision_eval" / "results" / "dual_camera_launcher"
POLL_INTERVAL = 0.5
POLL_SECONDS = 20.0
MAX_STATUS_BYTES = 64 * 1024


class LauncherError(Exception):
    pass


def _errmsg(kind):
    return {
        "password": "SSH 密码验证失败，请检查本地凭据文件。",
        "hostkey": "SSH 主机密钥校验失败，请确认 known_hosts 正确。",
        "ssh": "无法通过 SSH 连接 Ubuntu，请检查 Ubuntu、Tailscale 和 SSH 服务。",
        "preflight": "远端预检失败：脚本或摄像头状态不符合要求。",
        "launch": "远端启动预览服务失败。",
        "no_service": "未能在限定时间内连接到可用的预览服务。",
        "browser": "无法打开默认浏览器。",
        "config": "本地配置文件缺失或格式无效。",
    }.get(kind, "启动器发生错误。")


def _resolve(p):
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def load_config(path=None):
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        raise LauncherError("配置文件不存在：" + _errmsg("config")) from None
    except (OSError, ValueError):
        raise LauncherError("配置文件无法读取。" + _errmsg("config")) from None
    if not isinstance(cfg, dict):
        raise LauncherError("配置结构无效。" + _errmsg("config"))

    try:
        host = cfg["host"]
        ipaddress.ip_address(host)
        port = cfg["port"]
        assert isinstance(port, int) and not isinstance(port, bool)
        assert 1 <= port <= 65535
        username = cfg["username"]
        assert isinstance(username, str) and username.strip()
        sha = cfg["expected_source_sha256"]
        assert isinstance(sha, str) and len(sha) == 64
        assert all(c in "0123456789abcdefABCDEF" for c in sha)
        max_seconds = cfg["max_seconds"]
        assert isinstance(max_seconds, int) and not isinstance(max_seconds, bool)
        assert max_seconds > 0
        devices = cfg["camera_devices"]
        assert isinstance(devices, list) and len(devices) == 2
        assert len(set(devices)) == 2
        for d in devices:
            assert isinstance(d, str) and d.startswith("/dev/")
        for key in ("remote_python", "remote_script", "remote_lock",
                    "remote_log"):
            val = cfg[key]
            assert isinstance(val, str) and val.startswith("/")
        assert cfg["url"] == "http://{}:{}/".format(host, port)
    except (KeyError, TypeError, AssertionError, ValueError):
        raise LauncherError("配置字段无效。" + _errmsg("config")) from None

    for key in ("credential_file", "dependency_dir", "known_hosts"):
        val = cfg.get(key)
        assert isinstance(val, str) and val.strip(), key
        cfg[key] = str(_resolve(val))

    return cfg


def _direct_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe_status(config, timeout=2.0):
    req = urllib.request.Request(config["url"] + "status", method="GET")
    try:
        opener = _direct_opener()
        with opener.open(req, timeout=timeout) as resp:
            data = resp.read(MAX_STATUS_BYTES + 1)
        if len(data) > MAX_STATUS_BYTES:
            return None
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    cams = parsed.get("cameras")
    if not isinstance(cams, list) or len(cams) != 2:
        return None
    for idx, cam in enumerate(cams):
        if not isinstance(cam, dict):
            return None
        if cam.get("device") != config["camera_devices"][idx]:
            return None
        seq = cam.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            return None
        if "error" not in cam or not (cam["error"] is None or isinstance(cam["error"], str)):
            return None
        if "age_ms" not in cam:
            return None
        age = cam.get("age_ms")
        if age is not None:
            if isinstance(age, bool) or not isinstance(age, (int, float)):
                return None
            if not math.isfinite(age):
                return None
        for dim in ("width", "height"):
            v = cam.get(dim)
            if not isinstance(v, int) or isinstance(v, bool):
                return None
    return parsed


def is_ready(status):
    if not isinstance(status, dict):
        return False
    cams = status.get("cameras")
    if not isinstance(cams, list) or len(cams) != 2:
        return False
    for cam in cams:
        if not isinstance(cam, dict):
            return False
        seq = cam.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            return False
        if cam.get("error") is not None:
            return False
        if "age_ms" not in cam:
            return False
        age = cam.get("age_ms")
        if age is None or isinstance(age, bool):
            return False
        if not isinstance(age, (int, float)):
            return False
        if not math.isfinite(age) or not (0 <= age <= 2000):
            return False
        for dim in ("width", "height"):
            v = cam.get(dim)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                return False
    return True


def connect_ssh(config):
    dep = config.get("dependency_dir")
    if dep and dep not in sys.path:
        sys.path.insert(0, dep)
    import paramiko
    try:
        with open(config["credential_file"], "r", encoding="utf-8-sig") as fh:
            password = fh.read().rstrip("\r\n")
    except OSError:
        raise LauncherError("无法读取本地凭据文件。") from None
    if not password:
        raise LauncherError("本地凭据文件为空。")
    client = paramiko.SSHClient()
    try:
        try:
            client.load_host_keys(config["known_hosts"])
        except FileNotFoundError:
            raise LauncherError(_errmsg("hostkey")) from None
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            config["host"],
            username=config["username"],
            password=password,
            look_for_keys=False,
            allow_agent=False,
            timeout=8,
            auth_timeout=8,
            banner_timeout=8,
        )
    except paramiko.AuthenticationException:
        client.close()
        raise LauncherError(_errmsg("password")) from None
    except paramiko.BadHostKeyException:
        client.close()
        raise LauncherError(_errmsg("hostkey")) from None
    except paramiko.SSHException:
        client.close()
        raise LauncherError(_errmsg("ssh")) from None
    except LauncherError:
        try:
            client.close()
        except Exception:
            pass
        raise
    except Exception:
        client.close()
        raise LauncherError(_errmsg("ssh")) from None
    finally:
        password = None
    return client


def _run_remote(client, cmd_str):
    stdin, stdout, stderr = client.exec_command(cmd_str, timeout=12)
    out = stdout.read(8192)
    stderr.read(8192)
    rc = stdout.channel.recv_exit_status()
    return rc, out, b""


def _preflight(client, config):
    import shlex
    parts = ["test", "-x", shlex.quote(config["remote_python"]), "-a",
             "-r", shlex.quote(config["remote_script"])]
    for dev in config["camera_devices"]:
        parts += ["-a", "-e", shlex.quote(dev)]
    rc, _, _ = _run_remote(client, " ".join(parts))
    if rc != 0:
        raise LauncherError("远端预检失败：Python 或脚本或摄像头不可用。" + _errmsg("preflight"))

    rc, out, _ = _run_remote(
        client,
        "sha256sum -- " + shlex.quote(config["remote_script"]),
    )
    if rc != 0:
        raise LauncherError("远端预检失败：无法计算脚本哈希。" + _errmsg("preflight"))
    sha_line = out.decode("utf-8", "replace").strip()
    tokens = sha_line.split()
    if not tokens:
        raise LauncherError("远端预检失败：无法计算脚本哈希。" + _errmsg("preflight"))
    actual = tokens[0]
    if actual.lower() != config["expected_source_sha256"].lower():
        raise LauncherError("远端预检失败：脚本哈希已变化。" + _errmsg("preflight"))

    cam_a = shlex.quote(config["camera_devices"][0])
    cam_b = shlex.quote(config["camera_devices"][1])
    rc2, out2, _ = _run_remote(
        client, "realpath {}; realpath {}".format(cam_a, cam_b))
    if rc2 != 0:
        raise LauncherError("远端预检失败：摄像头路径无法解析。" + _errmsg("preflight"))
    lines = [ln.strip() for ln in out2.decode("utf-8", "replace").splitlines() if ln.strip()]
    if len(lines) != 2 or lines[0] == lines[1]:
        raise LauncherError("远端预检失败：两个摄像头必须不同。" + _errmsg("preflight"))


def start_remote(client, config):
    import shlex
    _preflight(client, config)
    remote_cmd = (
        "nohup flock -n {lock} {py} -u {script} --host {host} --port {port} "
        "--camera-a {ca} --camera-b {cb} --max-seconds {ms} >> {log} 2>&1 "
        "</dev/null & echo $!"
    ).format(
        lock=shlex.quote(config["remote_lock"]),
        py=shlex.quote(config["remote_python"]),
        script=shlex.quote(config["remote_script"]),
        host=shlex.quote(config["host"]),
        port=shlex.quote(str(config["port"])),
        ca=shlex.quote(config["camera_devices"][0]),
        cb=shlex.quote(config["camera_devices"][1]),
        ms=shlex.quote(str(config["max_seconds"])),
        log=shlex.quote(config["remote_log"]),
    )
    try:
        rc, out, _ = _run_remote(client, remote_cmd)
    except Exception:
        raise LauncherError(_errmsg("launch")) from None
    if rc != 0:
        raise LauncherError(_errmsg("launch"))
    pid_txt = out.decode("utf-8", "replace").strip()
    pid = None
    if pid_txt.isdigit():
        pid = int(pid_txt)
    if pid is None or pid <= 0:
        raise LauncherError(_errmsg("launch"))
    return {"pid": pid}


def ensure_preview(config, open_browser=True, notify=None):
    def _n(msg):
        if notify:
            try:
                notify(msg)
            except Exception:
                pass

    def _open():
        if open_browser:
            try:
                opened = webbrowser.open(config["url"], new=2)
            except Exception:
                raise LauncherError(_errmsg("browser")) from None
            if opened is False:
                raise LauncherError(_errmsg("browser"))

    status = probe_status(config)
    if status is not None and is_ready(status):
        _n("检测到已就绪的服务。")
        _open()
        return {
            "url": config["url"],
            "started": False,
            "ready": True,
            "status": status,
            "warning": None,
        }

    started = False
    if status is not None:
        _n("检测到已运行的服务。")
    else:
        _n("正在连接 Ubuntu...")
        client = connect_ssh(config)
        try:
            status = probe_status(config)
            if status is None:
                _n("正在远端启动预览服务...")
                start_remote(client, config)
                started = True
        finally:
            try:
                client.close()
            except Exception:
                pass

    if not (isinstance(status, dict) and is_ready(status)):
        deadline = time.monotonic() + POLL_SECONDS
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            status = probe_status(config)
            if isinstance(status, dict) and is_ready(status):
                break

    if not isinstance(status, dict):
        raise LauncherError(_errmsg("no_service"))

    ready = is_ready(status)
    warning = None
    if not ready:
        warning = "预览服务已启动，但至少一个摄像头未就绪，请打开页面检查。"

    _open()

    return {
        "url": config["url"],
        "started": started,
        "ready": ready,
        "status": status,
        "warning": warning,
    }


def _write_result(result, error=None):
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": error,
            "result": result,
        }
        with open(RESULTS_DIR / "last_run.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _messagebox(text):
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, "双摄像头预览启动器", 0x40)
    except Exception:
        pass


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="dual_camera_launcher_v2")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    try:
        ns, _unknown = parser.parse_known_args(
            list(sys.argv[1:] if argv is None else argv))
    except SystemExit:
        return 2

    try:
        config = load_config(ns.config)
        result = ensure_preview(config, open_browser=not ns.no_browser)
    except LauncherError as exc:
        msg = str(exc)
        _write_result(None, msg)
        if not ns.no_ui:
            _messagebox(msg + "（" + type(exc).__name__ + "）")
        if sys.stdout is not None:
            print(json.dumps({"error": msg}, ensure_ascii=False))
        return 1
    except Exception as exc:
        msg = _errmsg("ssh")
        _write_result(None, msg)
        if not ns.no_ui:
            _messagebox(msg)
        if sys.stdout is not None:
            print(json.dumps({"error": msg}, ensure_ascii=False))
        return 1

    _write_result(result)
    if not ns.no_ui and result.get("warning"):
        _messagebox(result["warning"])
    if ns.no_ui and sys.stdout is not None:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
