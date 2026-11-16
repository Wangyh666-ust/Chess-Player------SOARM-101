from collections import Counter


def summarize(windows, samples, squares, decode):
    local = {}
    reasons = {}
    candidates = []

    for square in squares:
        labels = [decode(rows[square].get("final")) for _, rows in windows]
        counts = Counter(label for label in labels if label is not None)
        vote_indexes = {i for i, label in enumerate(labels) if label is not None}
        observations = samples.get(square, ())
        observed_indexes = {idx for idx, _ in observations}

        main_count = 0
        main_label = None
        if counts:
            main_label, main_count = counts.most_common(1)[0]

        if main_count >= 2:
            local[square] = main_label
            reasons[square] = "%d/3 个窗口一致" % main_count
        elif counts or samples.get(square):
            local[square] = "unsure"
            reasons[square] = "三个窗口内未形成一致类别"
        else:
            local[square] = "empty"
            reasons[square] = "窗口内未检测到棋子"

        conflict_indexes = [idx for idx, conflict in observations if conflict]
        conflict_windows = set(conflict_indexes)

        evidence_windows = vote_indexes | observed_indexes
        duration = len(evidence_windows)

        candidate = None

        if local[square] == "unsure" and len(evidence_windows) >= 2:
            candidate = {
                "square": square,
                "priority": 1,
                "reason": "至少两个窗口有棋子迹象，类别未定",
            }
        elif len(conflict_indexes) >= 3 and len(conflict_windows) >= 2:
            duration = len(conflict_windows)
            candidate = {
                "square": square,
                "priority": 2,
                "reason": "两路在至少两个窗口持续分歧",
            }
        elif sum(counts.values()) == 3 and main_count == 2:
            candidate = {
                "square": square,
                "priority": 3,
                "reason": "三个窗口的类别投票为 2 比 1",
            }

        if candidate is not None:
            candidates.append((candidate, duration))

    candidates.sort(key=lambda item: (item[0]["priority"], -item[1], item[0]["square"]))

    return local, reasons, [candidate for candidate, _ in candidates]


def select_candidates(candidates, available):
    requests = {}
    selected = {}
    skipped = {}

    limit = 5

    for candidate in candidates:
        square = candidate["square"]
        pair = available.get(square)
        if pair is None:
            skipped[square] = "缺少可用双摄裁图"
            continue
        if len(requests) >= limit:
            skipped[square] = "超过本轮 5 格上限"
            continue
        requests[square] = pair
        selected[square] = candidate["reason"]

    return requests, selected, skipped
