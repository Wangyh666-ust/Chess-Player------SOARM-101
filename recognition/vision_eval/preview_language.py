"""Localize the data-collection preview page.

Exposes ``localize_preview(html)`` which inserts a language picker in the
header actions and an inline localization script before ``</body>``. All
other bytes of the input HTML are preserved.
"""

from preview_language_messages import MESSAGE_SCRIPT

HEADER_ACTIONS_MARKER = '<div class="actions">'
BODY_END_MARKER = "</body>"

PICKER_HTML = (
    '<label class="preview-language-label" for="preview-language">Language / 语言</label>'
    '<select id="preview-language" class="preview-language-select" autocomplete="off">'
    '<option value="zh-CN">中文</option>'
    '<option value="en">English</option>'
    '</select>'
)

_SCRIPT_TEMPLATE = MESSAGE_SCRIPT + r"""
(function () {
  "use strict";

  var STORAGE_KEY = "dualcam.preview.language";
  var DEFAULT_LANG = "zh-CN";
  var SUPPORTED = { "zh-CN": true, "en": true };

  // Simple exact-match labels (whitespace preserved around the match).
  var SIMPLE_LABELS = {
    "双摄数据采集": "Dual-camera preview",
    "预览 + 手动拍摄": "Live preview + snapshot",
    "模型: 未知": "Model: unknown",
    "摄像头2": "Camera 2",
    "摄像头1": "Camera 1",
    "主视角": "Primary view",
    "辅助视角": "Auxiliary view",
    "等待画面…": "Waiting for video…",
    "摄像头2画面": "Camera 2 feed",
    "摄像头1画面": "Camera 1 feed",
    "已暂停": "Paused",
    "旋转:": "Rotation:",
    "270°（逆时针 90°）": "270° (90° counterclockwise)",
    "显示 YOLO＋格线＋格子坐标": "Show YOLO, grid and cell coordinates",
    "暂停": "Pause",
    "继续": "Resume",
    "尺寸:": "Size:",
    "帧延迟:": "Frame age:",
    "推理:": "Inference:",
    "检测数:": "Detections:",
    "序号:": "Sequence:",
    "请求超时": "Request timed out",
    "视图请求失败": "Camera request failed",
    "保存目录（本地 Windows）": "Save folder (Windows)",
    "拍摄双摄图片": "Capture both views",
    "拍摄中…": "Capturing…",
    "下载配对 ZIP": "Download paired ZIP",
    "下载 ZIP": "Download ZIP",
    "拍摄请求超时（未自动重试）": "Capture timed out (no retry)",
    "棋局关联 · 相机2主视角 / 相机1辅助": "Board recognition · C2 primary / C1 auxiliary",
    "试运行：参考棋盘自动跟踪，底座位置为估计；未识别不等于空格。": "The grid is tracked automatically. Piece bases are estimated. A missed detection can hide a piece.",
    "棋局更新周期": "Board update interval",
    "Board update": "Board update interval",
    "不确定时：当前60% + 历史40%": "Uncertain estimates: 60% current + 40% history",
    "读取设置": "Loading settings…",
    "收集中": "Collecting:",
    "应用中": "Applying…",
    "已切换，重新收集": "Applied, collecting:",
    "双摄一致": "Both views agree",
    "冲突": "Conflict",
    "辅助待确认": "Auxiliary pending",
    "未知": "Unknown",
    "模型版本:": "Model version:",
    "状态:": "Status:",
    "加载失败": "Load failed",
    "已加载": "Loaded",
    "未加载": "Not loaded",
    "当前微调模型": "Fine-tuned model",
    "阶段耗时": "Pipeline timing",
    "Pipeline timing": "Pipeline timing",
    "格线管道：无数据": "Grid: no data",
    "格线管道：未启用（勾选后启用）": "Grid: disabled",
    "格线管道：错误": "Grid error:",
    "识别关闭": "recognition off",
    "格线耗时=": "grid time=",
    "定位中，YOLO暂停": "Localizing, YOLO paused",
    "缓存错误：": "Cache error:",
    "双摄时间差=": "camera time offset=",
    "辅助可用": "Auxiliary view available",
    "辅助未参与：等待两路新鲜且时间接近的结果": "Waiting for fresh, time-aligned views",
    "原始主摄": "raw primary",
    "原始辅摄": "raw auxiliary",
    "来源": "source",
    "多帧": "multi-frame",
    "权重 C2:C1": "weight C2:C1",
    "原因": "reason",
    "辅助拒绝": "auxiliary rejected",
    "无": "none",
    "识别棋局": "Recognize board",
    "重新识别棋局": "Recognize again",
    "启用 Qwen 辅助复核": "Enable Qwen review",
    "确认并定格": "Confirm and freeze",
    "复核详情": "Review details",
    "双摄裁图": "Camera crops",
    "保留待确认": "Leave unresolved",
    "空格": "Empty",
    "未确认": "Unresolved",
    "白兵": "White pawn",
    "白马": "White knight",
    "白象": "White bishop",
    "白车": "White rook",
    "白后": "White queen",
    "白王": "White king",
    "黑兵": "Black pawn",
    "黑马": "Black knight",
    "黑象": "Black bishop",
    "黑车": "Black rook",
    "黑后": "Black queen",
    "黑王": "Black king",
    "复核中": "Reviewing",
    "Qwen 未确认": "Qwen could not confirm",
    "投票未形成多数": "No majority",
    "本地未确认，未进入复核名单": "Unresolved locally and outside the review list",
    "Qwen 未能确认": "Qwen could not confirm",
    "复核结果不可用": "Review result unavailable",
    "Qwen 建议待确认": "Qwen left this unresolved",
    "缺少可用双摄裁图": "Camera crops unavailable",
    "超过本轮 5 格上限": "Outside the five-square review limit",
    "至少两个窗口有棋子迹象，类别未定": "Piece detected in at least two windows, class unresolved",
    "两路在至少两个窗口持续分歧": "Views disagree in at least two windows",
    "五个窗口的类别投票为 3 比 2": "Five-window vote split 3 to 2",
    "双摄预热超时。请确认 Ubuntu 的双摄预览可用、棋盘编码清晰可见。": "Dual-camera warm-up timed out. Check that the Ubuntu preview is available and the board code is clearly visible.",
    "双摄预览的投票周期需要设为 2 秒。": "The dual-camera preview vote interval must be set to 2 seconds.",
    "暂时无法读取棋局识别状态。": "Unable to read board recognition status right now.",
    "棋局状态已更新，请稍后再试。": "Board status was updated. Please try again shortly.",
    "棋局识别操作未完成。": "Board recognition did not complete."
  };

  // Multi-line / structured hint sentences.
  var HINT_1 = "只在点击拍摄时保存；直接保存到指定目录；文件名含时间戳与配对编号，不创建拍摄子文件夹。拍摄后自动保存到本机，无需手动下载。";
  var HINT_1_EN = "Saved only when you click capture. Files go directly into the chosen folder; names include a timestamp and pairing id, with no per-capture subfolder. Captures are stored locally automatically; no manual download is needed.";
  var HINT_2 = "从两个摄像头当前画面各截取一张，保存原图、旋转后图像与 YOLO 标注。暂停状态下仍按当前画面拍摄。";
  var HINT_2_EN = "Captures one frame from each camera's current view and saves the original image, the rotated image and the YOLO annotation. While paused it still captures the current frame.";

  var NOTICE_ZH = "本页面仅做实时预览与手动截图，不进行连续录制。勾选后显示格线与棋子坐标；相机2编号固定，参考跟踪失效时不输出格子。预测与底座归属均需人工复核。检测框坐标为旋转后图像像素；YOLO 预览较慢，显示的是模型预测结果，需人工复核。 两摄像头时间戳为各自的主机编码时间，非硬件同步。";
  var NOTICE_EN = "This page only provides live preview and manual snapshots; there is no continuous recording. When checked, the grid and piece coordinates are shown. Camera 2 numbering is fixed; when reference tracking fails no grid is output. Predictions and piece-base attribution require manual review. Detection-box coordinates are pixels in the rotated image; the YOLO preview is slow and shows model predictions that need manual review. Each camera timestamp is its own host-encoding time, not hardware-synchronized.";
  var NOTICE_LOCAL_ZH = "照片自动保存到当前 Windows 本机；Ubuntu 也保留拍摄备份。";
  var NOTICE_LOCAL_EN = "Photos are saved automatically to this Windows machine; Ubuntu also keeps a capture backup.";

  var SENTENCE_LABELS = {};
  SENTENCE_LABELS[HINT_1] = HINT_1_EN;
  SENTENCE_LABELS[HINT_2] = HINT_2_EN;
  SENTENCE_LABELS[NOTICE_ZH] = NOTICE_EN;
  SENTENCE_LABELS[NOTICE_LOCAL_ZH] = NOTICE_LOCAL_EN;

  function normalizeWhitespace(text) {
    return String(text).replace(/\s+/g, " ").trim();
  }

  // Store original (Chinese source) strings and the last value we rendered.
  var nodeState = new WeakMap();
  var attrState = new WeakMap();

  function getNodeState(node) {
    var st = nodeState.get(node);
    if (!st) {
      st = { source: null, rendered: null };
      nodeState.set(node, st);
    }
    return st;
  }

  function getAttrState(node) {
    var st = attrState.get(node);
    if (!st) {
      st = {};
      attrState.set(node, st);
    }
    return st;
  }

  var ATTRS = ["title", "alt", "aria-label"];

  function isSkippedElement(el) {
    var tag = el.tagName ? el.tagName.toLowerCase() : "";
    if (tag === "script" || tag === "style" || tag === "pre" || tag === "code" || tag === "textarea") {
      return true;
    }
    if (el.id === "preview-language") return true;
    if (el.classList && el.classList.contains("preview-language-label")) return true;
    if (el.classList && el.classList.contains("preview-language-select")) return true;
    return false;
  }

  function hasSkippedAncestor(node) {
    var cur = node.parentNode;
    while (cur) {
      if (cur.nodeType === 1) {
        if (isSkippedElement(cur)) return true;
      }
      cur = cur.parentNode;
    }
    return false;
  }

  function translateSimpleText(text) {
    if (Object.prototype.hasOwnProperty.call(SIMPLE_LABELS, text)) {
      return SIMPLE_LABELS[text];
    }
    return null;
  }

  function translateSentenceText(text) {
    var norm = normalizeWhitespace(text);
    if (Object.prototype.hasOwnProperty.call(SENTENCE_LABELS, norm)) {
      return SENTENCE_LABELS[norm];
    }
    return null;
  }

  function localizedValue(value, state, lang) {
    if (value !== state.rendered) state.source = value;
    var source = state.source;
    var text = source.trim();
    var english = lang === "en" && text ? (translateSimpleText(text) ??
      translateSentenceText(text) ?? englishUiMessage(text, SIMPLE_LABELS)) : null;
    state.rendered = english === null ? source : source.replace(text, function () { return english; });
    return state.rendered;
  }

  function processTextNode(node, lang) {
    if (!node || node.nodeType !== 3) return;
    if (!node.isConnected) return;
    if (hasSkippedAncestor(node)) return;

    var value = node.nodeValue;
    if (value === null || value === undefined) return;
    if (value.trim() === "") return;

    var next = localizedValue(value, getNodeState(node), lang);
    if (next !== value) node.nodeValue = next;
  }

  function processAttribute(node, attr, lang) {
    if (!node.isConnected) return;
    var current = node.getAttribute(attr);
    if (current === null || current === undefined) return;

    var stMap = getAttrState(node);
    var st = stMap[attr];
    if (!st) {
      st = { source: null, rendered: null };
      stMap[attr] = st;
    }

    var next = localizedValue(current, st, lang);
    if (next !== current) node.setAttribute(attr, next);
  }

  function processElementTree(root, lang) {
    if (!root) return;
    if (root.nodeType === 3) {
      processTextNode(root, lang);
      return;
    }
    if (root.nodeType !== 1) {
      // Document / fragment: recurse children.
      var kids = root.childNodes;
      if (kids) {
        for (var i = 0; i < kids.length; i++) processElementTree(kids[i], lang);
      }
      return;
    }
    if (isSkippedElement(root)) return;

    for (var a = 0; a < ATTRS.length; a++) {
      processAttribute(root, ATTRS[a], lang);
    }

    var children = root.childNodes;
    for (var c = 0; c < children.length; c++) {
      processElementTree(children[c], lang);
    }
  }

  function currentLanguage() {
    try {
      var stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored && SUPPORTED[stored]) return stored;
    } catch (e) {
      // ignore storage errors
    }
    return DEFAULT_LANG;
  }

  function persistLanguage(lang) {
    try {
      window.localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) {
      // ignore storage errors
    }
  }

  var activeLang = currentLanguage();

  function applyLanguage(lang, root) {
    activeLang = lang;
    if (document.documentElement) {
      document.documentElement.setAttribute("lang", lang === "en" ? "en" : "zh-CN");
    }
    processElementTree(root || document, lang);
  }

  function setupPicker() {
    var picker = document.getElementById("preview-language");
    if (!picker) return;
    picker.value = activeLang;
    picker.addEventListener("change", function () {
      var val = picker.value;
      if (!SUPPORTED[val]) return;
      persistLanguage(val);
      applyLanguage(val);
    });
  }

  function startObserver() {
    if (typeof MutationObserver === "undefined") return;
    var observer = new MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        var rec = records[i];
        if (rec.type === "characterData") {
          processTextNode(rec.target, activeLang);
        } else if (rec.type === "childList") {
          for (var j = 0; j < rec.addedNodes.length; j++) {
            var added = rec.addedNodes[j];
            if (added.isConnected) processElementTree(added, activeLang);
          }
        } else if (rec.type === "attributes") {
          var el = rec.target;
          if (el && el.isConnected && !isSkippedElement(el)) {
            processAttribute(el, rec.attributeName, activeLang);
          }
        }
      }
    });
    observer.observe(document, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: ATTRS
    });
  }

  setupPicker();
  applyLanguage(activeLang);
  startObserver();
})();
"""

_SCRIPT_HTML = "<script>\n" + _SCRIPT_TEMPLATE + "\n</script>\n"


def localize_preview(html: str) -> str:
    """Return ``html`` with the language picker and localization script inserted."""
    if not isinstance(html, str):
        raise TypeError("html must be a string")

    if html.count(HEADER_ACTIONS_MARKER) != 1:
        raise ValueError(
            "expected exactly one '<div class=\"actions\">' occurrence"
        )
    if html.count(BODY_END_MARKER) != 1:
        raise ValueError("expected exactly one '</body>' occurrence")

    html = html.replace(HEADER_ACTIONS_MARKER, HEADER_ACTIONS_MARKER + '\n    ' + PICKER_HTML, 1)
    return html.replace(BODY_END_MARKER, _SCRIPT_HTML + BODY_END_MARKER, 1)
