(() => {
  'use strict';
  const get = id => document.getElementById(id);
  const start = get('board-review-start');
  const useQwen = get('board-review-use-qwen');
  const confirm = get('board-review-confirm');
  const status = get('board-review-status');
  const details = get('board-review-details');
  const rows = get('board-review-rows');
  const board = get('grid-board');
  const names = {wP:'白兵', wN:'白马', wB:'白象', wR:'白车', wQ:'白后', wK:'白王',
    bP:'黑兵', bN:'黑马', bB:'黑象', bR:'黑车', bQ:'黑后', bK:'黑王', empty:'空格', unsure:'未确认'};
  const active = new Set(['warmup', 'collecting', 'reviewing']);
  let snapshot, viewKey, editingRun, awaiting = false, requestError = '';
  let manual = {};

  function shownDraft() {
    return ['reviewing', 'ready'].includes(snapshot?.phase) ? snapshot.draft : null;
  }
  function frozenBoard() {
    return snapshot?.phase === 'confirmed' ? snapshot.confirmed : null;
  }
  window.boardReview = {ownsBoard: () => !!(shownDraft() || frozenBoard())};

  function renderBoard() {
    const draft = shownDraft();
    const frozen = frozenBoard();
    const mapping = draft?.local || frozen?.board;
    if (!mapping) return;
    board.replaceChildren();
    for (let rank = 8; rank >= 1; rank--) {
      for (const [file, letter] of Array.from('abcdefgh').entries()) {
        const sq = letter + rank;
        const original = mapping[sq];
        const opinion = draft?.opinions?.[sq];
        const changed = opinion?.label && opinion.label !== 'unsure' && opinion.label !== original;
        const label = manual[sq] || (changed ? opinion.label : original);
        const unresolved = !manual[sq] && (draft?.manual?.[sq] || frozen?.unresolved?.[sq]);
        const cell = document.createElement('div');
        cell.className = `grid-sq ${(file + rank) % 2 ? 'dark' : 'light'}`;
        cell.classList.toggle('review-inflight', snapshot.phase === 'reviewing' && !!draft.selected[sq]);
        cell.classList.toggle('review-corrected', !!(changed || manual[sq] || frozen?.manual?.[sq] || frozen?.accepted?.includes(sq)));
        cell.classList.toggle('review-manual', !!unresolved);
        cell.title = [sq, names[label], changed && `Qwen 修正 ${names[original]} → ${names[label]}`,
          unresolved].filter(Boolean).join(' · ');
        const square = document.createElement('span');
        const piece = document.createElement('span');
        square.className = 'sq-name';
        square.textContent = sq;
        piece.className = 'sq-piece';
        piece.textContent = label === 'empty' ? '' : label === 'unsure' ? '?' : label?.toUpperCase() || '';
        cell.append(square, piece);
        board.append(cell);
      }
    }
  }

  function makeRow(draft, sq, reason) {
    const row = document.createElement('div');
    row.className = 'review-row';
    const opinion = draft.opinions[sq];
    const sent = Object.hasOwn(draft.selected, sq);
    const text = document.createElement('span');
    const result = opinion?.label === 'unsure' ? 'Qwen 未确认' : opinion && `Qwen ${names[opinion.label]}`;
    text.textContent = [sq, names[draft.local[sq]], result || (sent && snapshot.phase === 'reviewing' ? '复核中' : ''),
      reason, draft.manual[sq] === reason ? '' : draft.manual[sq]].filter(Boolean).join(' · ');
    text.title = [opinion?.C1 && `C1 ${opinion.C1}`, opinion?.C2 && `C2 ${opinion.C2}`].filter(Boolean).join(' | ');
    row.append(text);
    if (snapshot.phase === 'ready' && draft.manual[sq]) {
      const select = document.createElement('select');
      select.setAttribute('aria-label', `${sq} 人工确认`);
      select.add(new Option('保留待确认', ''));
      for (const [value, name] of Object.entries(names)) {
        if (value !== 'unsure') select.add(new Option(name, value));
      }
      select.value = manual[sq] || '';
      select.addEventListener('change', () => {
        if (select.value) manual[sq] = select.value;
        else delete manual[sq];
        renderBoard();
      });
      row.append(select);
    }
    if (snapshot.phase === 'ready' && sent) {
      const imagePanel = document.createElement('details');
      const open = document.createElement('summary');
      open.textContent = '双摄裁图';
      imagePanel.append(open);
      for (const camera of ['C1', 'C2']) {
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.alt = `${sq} ${camera}`;
        img.src = `/board-review/crop/${draft.id}/${sq}/${camera}`;
        imagePanel.append(img);
      }
      row.append(imagePanel);
    }
    return row;
  }

  function render() {
    if (!snapshot) return;
    const {phase, seconds, capture_seconds, review_seconds} = snapshot;
    const draft = shownDraft();
    const frozen = frozenBoard();
    const manualFlags = draft?.manual || frozen?.unresolved || {};
    const total = Object.keys(draft?.selected || {}).length;
    const unresolved = Object.keys(manualFlags).filter(sq => !manual[sq]);
    const corrected = Object.entries(draft?.opinions || {}).filter(([sq, opinion]) =>
      opinion.label !== 'unsure' && opinion.label !== draft.local[sq]).length;
    const messages = {
      idle: '采集 10 秒，汇总 5 个双摄投票结果。',
      warmup: '正在等待两路相机和棋盘定位。',
      collecting: `正在采集 ${Number(seconds).toFixed(1)} / ${capture_seconds ?? 10} 秒，请保持棋盘静止。`,
      reviewing: `已显示双摄融合棋局，Qwen 正在复核 ${total} 格（${Math.floor(review_seconds || 0)} 秒）。`,
      ready: draft && !draft.use_qwen ? `10 秒投票完成。${unresolved.length ? `${unresolved.length} 格未形成多数。` : '已显示最终棋局。'}` :
        draft?.review_error ? `${draft.review_error}。${unresolved.length} 格待人工查看。` :
        total ? `已复核 ${total} 格，自动修正 ${corrected} 格，${unresolved.length} 格待人工查看。` :
          `已完成双摄融合。${unresolved.length ? `${unresolved.length} 格待人工查看。` : '本轮无需复核。'}`,
      confirmed: unresolved.length ? `棋局已定格。待确认格 ${unresolved.join('、')}。` : '棋局已定格。',
      error: snapshot.message,
    };
    status.textContent = requestError || messages[phase];
    start.disabled = awaiting || active.has(phase);
    useQwen.disabled = start.disabled;
    confirm.disabled = awaiting;
    confirm.hidden = phase !== 'ready';
    start.textContent = ['idle', 'warmup', 'collecting'].includes(phase) ? '识别棋局' : '重新识别棋局';
    const key = `${phase}:${draft?.id || frozen?.id || ''}`;
    if (key === viewKey) return;
    viewKey = key;
    if (editingRun !== draft?.id) { manual = {}; editingRun = draft?.id; }
    renderBoard();
    rows.replaceChildren();
    const listed = new Map((draft?.queue || []).map(row => [row.square, row.reason]));
    for (const sq of Object.keys(manualFlags)) if (!listed.has(sq) && draft) listed.set(sq, draft.manual[sq]);
    details.hidden = listed.size === 0;
    get('board-review-summary').textContent = `复核详情（${listed.size} 格）`;
    for (const [sq, reason] of listed) rows.append(makeRow(draft, sq, reason));
  }

  async function refresh() {
    const reply = await fetch('/board-review/state', {cache:'no-store'});
    if (!reply.ok) throw new Error('暂时无法读取棋局识别状态。');
    snapshot = await reply.json();
    render();
  }
  async function send(action, body) {
    if (awaiting) return;
    awaiting = true;
    requestError = '';
    render();
    try {
      const response = await fetch(`/board-review/${action}`, {method:'POST', headers:window.__votingHeaders,
        body:JSON.stringify(body)});
      if (!response.ok) throw new Error(response.status === 409 ? '棋局状态已更新，请稍后再试。' : '棋局识别操作未完成。');
      await refresh();
    } catch (error) {
      requestError = error.message;
    } finally {
      awaiting = false;
      render();
    }
  }
  start.addEventListener('click', () => send('start', {use_qwen:useQwen.checked}));
  confirm.addEventListener('click', () => send('confirm', {id:snapshot.draft.id, manual}));
  async function poll() {
    try { if (!awaiting) await refresh(); }
    catch { status.textContent = '暂时无法读取棋局识别状态。'; }
    window.setTimeout(poll, 450);
  }
  poll();
})();
