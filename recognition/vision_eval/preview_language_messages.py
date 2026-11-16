"""JavaScript message translator for the board UI localization."""

MESSAGE_SCRIPT = r"""function englishUiMessage(text, labels) {
  if (typeof text !== "string") return null;
  const t = text.trim();
  if (!t) return null;
  const L = labels || {};
  const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  const sq = (n) => (n === 1 ? "square" : "squares");
  const label = (key) => hasOwn(L, key) ? L[key] : key;

  // ---- fixed phrases -------------------------------------------------
  if (t === "棋局已定格。") return "Board frozen.";
  if (t === "已完成双摄融合。本轮无需复核。") return "Dual-camera fusion complete. No review needed this round.";
  if (t === "本轮无需复核。") return "No review needed this round.";
  if (t === "正在等待两路相机和棋盘定位。") return "Waiting for both cameras and board localization.";
  if (t === "采集 10 秒，汇总 5 个双摄投票结果。") return "Collect 10 seconds of footage and combine five dual-camera votes.";

  // ---- phase / status sentences -------------------------------------
  let m;

  // 正在采集 X / Y 秒，请保持棋盘静止。
  m = t.match(/^正在采集\s+(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)\s*秒，请保持棋盘静止。$/);
  if (m) return `Collecting ${Number(m[1]).toFixed(1)} / ${m[2]} s. Keep the board still.`;

  // 已显示双摄融合棋局，Qwen 正在复核 N 格（M 秒）。
  m = t.match(/^已显示双摄融合棋局，Qwen 正在复核\s+(\d+)\s*格（(\d+)\s*秒）。$/);
  if (m) return `Dual-camera board displayed. Qwen is reviewing ${m[1]} ${sq(Number(m[1]))} (${m[2]} s).`;

  // 10 秒投票完成。+ variants
  m = t.match(/^10 秒投票完成。(.*)$/);
  if (m) {
    const tail = m[1];
    if (tail === "已显示最终棋局。") return "10-second vote complete. Final board displayed.";
    let mm = tail.match(/^(\d+)\s*格未形成多数。$/);
    if (mm) return `10-second vote complete. ${mm[1]} ${sq(Number(mm[1]))} ${Number(mm[1]) === 1 ? "has" : "have"} no majority.`;
  }

  // review_error + N 格待人工查看。
  // Handled via generic suffix below.

  // 已复核 N 格，自动修正 M 格，K 格待人工查看。
  m = t.match(/^已复核\s+(\d+)\s*格，自动修正\s+(\d+)\s*格，(\d+)\s*格待人工查看。$/);
  if (m) {
    return `Reviewed ${m[1]} ${sq(Number(m[1]))}; auto-corrected ${m[2]}; ${m[3]} ${sq(Number(m[3]))} ${Number(m[3]) === 1 ? "needs" : "need"} manual review.`;
  }

  // 已完成双摄融合。+ unresolved sentence or none
  m = t.match(/^已完成双摄融合。(.*)$/);
  if (m) {
    const tail = m[1];
    if (tail === "本轮无需复核。") return "Dual-camera fusion complete. No review needed this round.";
    let mm = tail.match(/^(\d+)\s*格待人工查看。$/);
    if (mm) return `Dual-camera fusion complete. ${mm[1]} ${sq(Number(mm[1]))} ${Number(mm[1]) === 1 ? "needs" : "need"} manual review.`;
  }

  // 棋局已定格。待确认格 c7、f4。
  m = t.match(/^棋局已定格。待确认格\s+(.+)。$/);
  if (m) return `Board frozen. Unresolved: ${m[1].replaceAll("、", ", ")}.`;

  // 网格汇总复核详情（N 格）
  m = t.match(/^复核详情（(\d+)\s*格）$/);
  if (m) return `Review details (${m[1]} ${sq(Number(m[1]))})`;

  // c7 人工确认
  m = t.match(/^([a-zA-Z]\d+)\s+人工确认$/);
  if (m) return `Manual review for ${m[1]}`;

  // suffix: N 格待人工查看。
  m = t.match(/^(.+?)。?(\d+)\s*格待人工查看。$/);
  if (m) {
    const prefix = m[1];
    const n = m[2];
    const suffix = `${n} ${sq(Number(n))} ${Number(n) === 1 ? "needs" : "need"} manual review.`;
    const translatedPrefix = translateProviderError(prefix);
    if (translatedPrefix) return `${translatedPrefix}. ${suffix}`;
    // unknown prefix: leave intact
    return null;
  }

  // 未确认，请看状态 / <suffix>
  m = t.match(/^未确认，请看状态 \/ Not confirmed: (.+)$/);
  if (m) return `Not confirmed: ${m[1]}`;

  // ---- common display messages ---------------------------------------
  m = t.match(/^已保存到本地：(.+)$/);
  if (m) return `Saved to: ${m[1]}`;

  m = t.match(/^拍摄失败:\s*(.+)$/);
  if (m) return `Capture failed: ${m[1]}`;

  m = t.match(/^默认:\s*(.+)$/);
  if (m) return `Default: ${m[1]}`;

  m = t.match(/^(\d+)\s*秒\s*\/\s*s$/);
  if (m) return `${m[1]} s`;

  m = t.match(/^(\d+)°（顺时针\s*(\d+)°）$/);
  if (m) return `${m[1]}° (${m[2]}° clockwise)`;

  if (t === "棋局更新周期 / Board update") return "Board update interval";
  if (t === "不确定时：当前60% + 历史40% / Uncertain: current 60% + previous 40%")
    return "Uncertain estimates: 60% current + 40% history";
  if (t === "读取设置 / Loading") return "Loading settings…";
  if (t === "应用中 / Applying…") return "Applying…";
  if (t === "阶段耗时 / Pipeline timing") return "Pipeline timing";

  m = t.match(/^上个窗口 \/ Last:\s*(\d+)\s*帧\s*\/\s*frames\s*·\s*(\d+(?:\.\d+)?)\s*s$/);
  if (m) return `Last window: ${m[1]} frames · ${m[2]} s`;

  m = t.match(/^收集中 \/ Collecting:\s*(\d+)\s*帧\s*\/\s*frames$/);
  if (m) return `Collecting: ${m[1]} frames`;

  m = t.match(/^已切换，重新收集 \/ Applied, collecting:\s*(\d+(?:\.\d+)?)\s*s$/);
  if (m) return `Applied, collecting: ${m[1]} s`;

  // ---- grid diagnostics ----------------------------------------------
  const diagPrefix = "7:3＋多帧试用 | C1原始检测；C2与下方棋盘为稳定结果（分数非正确率） | ";
  const diagEnPrefix = "7:3 with temporal aggregation | C1 raw detections | C2 and the board show stabilized predictions | ";
  if (t.startsWith(diagPrefix)) {
    const chunks = t.slice(diagPrefix.length).split(" | ");
    return diagEnPrefix + chunks.map((part) => {
      if (/^C[12]\b/.test(part)) {
        return part.replace(/^(C[12]) 暂停/, "$1 paused")
          .replace(/^(C[12]) 识别关闭/, "$1 recognition off")
          .replace(" 格线耗时=", " grid time=");
      }
      for (const prefix of ["格线管道：错误", "缓存错误：", "双摄时间差="]) {
        if (part.startsWith(prefix)) return label(prefix) + part.slice(prefix.length);
      }
      return label(part);
    }).join(" | ");
  }

  // ---- cell tooltip separated by ' | ' --------------------------------
  if (t.indexOf(" | ") !== -1) {
    const parts = t.split(" | ");
    const out = parts.map((raw) => {
      const p = raw.trim();
      if (!p) return p;
      if (p === "未知") return "Unknown";
      if (p === "无") return "None";
      if (hasOwn(L, p)) return L[p];
      for (const head of ["原始主摄", "原始辅摄", "来源", "多帧", "权重 C2:C1", "原因", "辅助拒绝"]) {
        if (p.startsWith(head + " ")) return label(head) + " " + label(p.slice(head.length + 1));
      }
      return raw;
    });
    return out.join(" | ");
  }

  // ---- ' · ' sections --------------------------------------------------
  if (t.indexOf(" · ") !== -1) {
    const parts = t.split(" · ");
    let changed = false;
    const out = parts.map((raw) => {
      const p = raw.trim();
      if (!p) return raw;
      if (hasOwn(L, p)) { changed = true; return L[p]; }
      // Qwen <Chinese piece>
      let mm = p.match(/^Qwen\s+([\u4e00-\u9fff]+)$/);
      if (mm && hasOwn(L, mm[1])) { changed = true; return `Qwen ${label(mm[1])}`; }
      // Qwen 修正 <piece> → <piece>
      mm = p.match(/^Qwen\s+修正\s+(.+?)\s*→\s*(.+)$/);
      if (mm) { changed = true; return `Qwen changed ${label(mm[1])} → ${label(mm[2])}`; }
      return raw;
    });
    return changed ? out.join(" · ") : null;
  }

  // ---- known board errors --------------------------------------------
  if (t === "采集结果缺少有效的 64 格棋盘。") return "Capture result is missing a valid 64-square board.";
  if (t === "复核格超过本轮 5 格上限。") return "Review cells exceed the 5-cell limit for this round.";
  if (t === "投票窗口的时间或帧数不完整。") return "Voting window timing or frame count is incomplete.";
  m = t.match(/^只收齐\s*(\d+)\/(\d+)\s*个连续投票窗口，请重新识别。$/);
  if (m) return `Only ${m[1]}/${m[2]} consecutive voting windows collected; please recognize again.`;

  return null;

  // ---- provider error translator --------------------------------------
  function translateProviderError(prefix) {
    const p = prefix.trim();
    if (!p) return null;
    const known = [
      ["本地接口预检失败", "Local endpoint preflight failed"],
      ["棋子参考照不可用", "Piece reference photo unavailable"],
      ["本机的阿里云凭据缺失或格式有误", "Local Aliyun credentials missing or malformed"],
      ["Qwen 未接受当前凭据或该模型权限不足", "Qwen rejected the current credentials or lacks model permission"],
      ["Qwen 响应超时", "Qwen response timed out"],
      ["暂时无法连接 Qwen 接口", "Temporarily unable to connect to the Qwen endpoint"],
      ["Qwen 拒绝了图片请求格式", "Qwen rejected the image request format"],
      ["Qwen 暂时限制了请求频率或额度", "Qwen temporarily rate-limited or quota-limited"],
      ["Qwen 回复格式未通过校验", "Qwen response failed format validation"],
      ["Qwen 未生成完整回复", "Qwen did not generate a complete response"],
      ["接口未返回有效结果", "Endpoint did not return a valid result"],
    ];
    for (const [zh, en] of known) {
      if (p === zh) return en;
    }
    // allow known message with extra punctuation
    const stripped = p.replace(/[。.!！?？]+$/, "");
    for (const [zh, en] of known) {
      if (stripped === zh) return en;
    }
    return null;
  }
}"""
