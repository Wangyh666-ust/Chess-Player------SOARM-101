"""Launch the Windows camera preview with board review on the same listener."""

from pathlib import Path
import hashlib
import sys

import local_capture_bridge_v9 as bridge
from preview_review_http import install
from preview_language import localize_preview


HERE = Path(__file__).resolve()
BOARD_DIR = HERE.parent / "board_confirmation"
PANEL = HERE.with_name("preview_review_panel.html")
WIDGET = HERE.with_name("preview_review_panel.js")
VENV = HERE.parents[1] / ".venv-chess-benchmark" / "Scripts"


def configure():
    sys.path.insert(0, str(BOARD_DIR))
    rewrite_preview = bridge._rewrite_ui
    bridge._HERE = HERE

    def source_sha256():
        files = (HERE, PANEL, WIDGET, HERE.with_name("preview_language.py"),
                 HERE.with_name("preview_language_messages.py"), HERE.with_name("local_capture_bridge_v9.py"),
                 HERE.with_name("preview_review_http.py"), BOARD_DIR / "server_budget.py",
                 BOARD_DIR / "qwen_live_budget_v2.py", BOARD_DIR / "live_capture_budget.py",
                 BOARD_DIR / "live_review_queue_v2.py", BOARD_DIR / "five_window_vote.py")
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def child_python():
        return str(VENV / "pythonw.exe") if sys.platform == "win32" else sys.executable

    def rewrite(html, token):
        html = rewrite_preview(html, token)
        panel = PANEL.read_text(encoding="utf-8")
        html = bridge._replace_once(html, '<div class="grid-health" id="grid-health"></div>',
                                    panel + '\n  <div class="grid-health" id="grid-health"></div>', "review controls")
        widget = '<script>\n' + WIDGET.read_text(encoding="utf-8") + '\n</script>\n'
        html = bridge._replace_once(html, "</body>", widget + "</body>", "review script")
        anchor = "    if(!healthEl || !boardEl) return;"
        html = bridge._replace_once(html, anchor, anchor +
            "\n    if(window.boardReview && window.boardReview.ownsBoard()) return;", "shared board renderer")
        return localize_preview(html)

    bridge._source_sha256 = source_sha256
    argv = sys.argv[1:]
    if "--serve" not in argv:
        bridge._python_for_child = child_python
    bridge._rewrite_ui = rewrite
    install(bridge)
    return bridge


if __name__ == "__main__":
    raise SystemExit(configure().main())
