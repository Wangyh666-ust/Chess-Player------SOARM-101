from collections import Counter

def summarize(windows, samples, squares, decode):
    local = {}
    reasons = {}
    queue = []

    for square in squares:
        labels = [decode(rows[square].get("final")) for _, rows in windows]
        votes = [label for label in labels if label is not None]
        counts = Counter(votes)

        winner, win_count = counts.most_common(1)[0] if counts else (None, 0)
        observations = samples.get(square, ())
        sample_idx = {index for index, _ in observations}

        vote_idx = {i for i, label in enumerate(labels) if label is not None}
        evidence_idx = vote_idx | sample_idx
        evidence_count = len(evidence_idx)

        if win_count >= 3:
            local[square] = winner
            reasons[square] = "%d/5 个窗口一致" % win_count
        else:
            local[square] = "unsure" if (votes or sample_idx) else "empty"
            reasons[square] = "五个窗口内未形成一致类别" if (votes or sample_idx) else "窗口内未检测到棋子"

        conflicts = [index for index, disagreed in observations if disagreed]
        conflict_windows = len(set(conflicts))
        if local[square] == "unsure" and evidence_count >= 2:
            queue.append((1, -evidence_count, square, "至少两个窗口有棋子迹象，类别未定"))
        elif len(conflicts) >= 3 and conflict_windows >= 2:
            queue.append((2, -conflict_windows, square, "两路在至少两个窗口持续分歧"))
        elif sorted(counts.values()) == [2, 3]:
            queue.append((3, -5, square, "五个窗口的类别投票为 3 比 2"))

    candidates = [dict(square=square, priority=priority, reason=reason)
                  for priority, _, square, reason in sorted(queue)]
    return local, reasons, candidates
