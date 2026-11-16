import review
import review_v3
from review_v3 import _clean_text

MAX_TOKENS = 3600

LABELS = review_v3.LABELS
BASE_VALUES = review_v3.BASE_VALUES
TOP_VALUES = review_v3.TOP_VALUES
TOP_TO_TYPE = review_v3.TOP_TO_TYPE

_ROW_KEYS = {"square", "observations", "label", "alternative", "evidence"}
_VIEW_KEYS = {"base", "footing", "waist", "silhouette", "top", "detail", "issue"}
_ISSUE_VALUES = {"clear", "glare", "blur", "occluded", "out_of_frame"}


def build_packet(prompt, requests, reference):
    packet = review_v3.build_packet(prompt, requests, reference)
    packet["max_tokens"] = MAX_TOKENS
    return packet


def _maybe_text(value, name, max_len=None):
    if value is None:
        return None
    return _clean_text(value, name, max_len)


def _parse_view(square_name, view_name, view):
    if not isinstance(view, dict) or set(view.keys()) != _VIEW_KEYS:
        raise ValueError("%s/%s must have exact keys %s" % (square_name, view_name, sorted(_VIEW_KEYS)))

    base = view["base"]
    if not isinstance(base, str) or base not in BASE_VALUES:
        raise ValueError("%s/%s base invalid" % (square_name, view_name))

    top = view["top"]
    if not isinstance(top, str) or top not in TOP_VALUES:
        raise ValueError("%s/%s top invalid" % (square_name, view_name))

    issue = view["issue"]
    if not isinstance(issue, str) or issue not in _ISSUE_VALUES:
        raise ValueError("%s/%s issue invalid" % (square_name, view_name))

    footing = _maybe_text(view["footing"], "%s/%s footing" % (square_name, view_name), 60)
    waist = _maybe_text(view["waist"], "%s/%s waist" % (square_name, view_name), 60)
    silhouette = _maybe_text(view["silhouette"], "%s/%s silhouette" % (square_name, view_name), 60)
    detail = _maybe_text(view["detail"], "%s/%s detail" % (square_name, view_name), 60)

    if base == "empty":
        if any(v is not None for v in (footing, waist, silhouette, detail)):
            raise ValueError("%s/%s empty base requires null morphology" % (square_name, view_name))
        if top != "unclear":
            raise ValueError("%s/%s empty base requires unclear top" % (square_name, view_name))
    else:
        if top != "unclear" and detail is None:
            raise ValueError("%s/%s identified top requires detail" % (square_name, view_name))

    return {
        "base": base,
        "footing": footing,
        "waist": waist,
        "silhouette": silhouette,
        "top": top,
        "detail": detail,
        "issue": issue,
    }


def _parse_row(square_name, row):
    if not isinstance(row, dict) or set(row.keys()) != _ROW_KEYS:
        raise ValueError("%s row must have exact keys %s" % (square_name, sorted(_ROW_KEYS)))

    square = row["square"]
    if square != square_name:
        raise ValueError("%s row square mismatch: %r" % (square_name, square))

    observations = row["observations"]
    if not isinstance(observations, dict) or set(observations.keys()) != {"C1", "C2"}:
        raise ValueError("%s observations must have exactly C1,C2" % square_name)

    c1 = _parse_view(square_name, "C1", observations["C1"])
    c2 = _parse_view(square_name, "C2", observations["C2"])

    label = row["label"]
    if not isinstance(label, str) or (label not in LABELS and label not in ("unsure", "empty")):
        raise ValueError("%s label invalid" % square_name)

    alternative = row["alternative"]
    if alternative is not None:
        if alternative not in LABELS:
            raise ValueError("%s alternative invalid" % square_name)
        if alternative == label:
            raise ValueError("%s alternative equals label" % square_name)

    if label == "empty" and alternative is not None:
        raise ValueError("%s empty label requires null alternative" % square_name)
    if label in LABELS and alternative is None:
        raise ValueError("%s piece label requires an alternative" % square_name)

    evidence = _clean_text(row["evidence"], "%s evidence" % square_name, 160)

    return {
        "square": square,
        "observations": {"C1": c1, "C2": c2},
        "label": label,
        "alternative": alternative,
        "evidence": evidence,
    }


def _visible_morphology_regions(parsed):
    """Return set of morphology region names supported by visible confirmed-target views."""
    regions = set()
    for view_name in ("C1", "C2"):
        view = parsed["observations"][view_name]
        if view["base"] != "target":
            continue
        if view["footing"] is not None:
            regions.add("footing")
        if view["waist"] is not None:
            regions.add("waist")
        if view["silhouette"] is not None:
            regions.add("silhouette")
        if view["top"] != "unclear" and view["detail"] is not None:
            regions.add("detail")
    return regions


def _consistency(parsed):
    """Apply conservative consistency checks for a piece label. Mutates parsed label."""
    label = parsed["label"]
    if label == "unsure":
        return

    if label == "empty":
        views = [parsed["observations"]["C1"], parsed["observations"]["C2"]]
        has_empty = any(v["base"] == "empty" for v in views)
        has_target = any(v["base"] == "target" for v in views)
        if not has_empty or has_target:
            parsed["guard_reason"] = "空格判定与两路观察不一致"
            parsed["label"] = "unsure"
        return

    views = [parsed["observations"]["C1"], parsed["observations"]["C2"]]

    if any(v["base"] == "empty" for v in views):
        parsed["guard_reason"] = "有一路观察到目标格为空"
        parsed["label"] = "unsure"
        return

    if not any(v["base"] == "target" for v in views):
        parsed["guard_reason"] = "两路均无法确认棋子落在目标格"
        parsed["label"] = "unsure"
        return

    label_type = label[1]
    target_views = [v for v in views if v["base"] == "target"]
    for view in target_views:
        top = view["top"]
        if top == "unclear":
            continue
        expected_type = TOP_TO_TYPE.get(top)
        if expected_type != label_type:
            parsed["guard_reason"] = "已定位棋子的顶部特征与所报类别冲突"
            parsed["label"] = "unsure"
            return

    regions = _visible_morphology_regions(parsed)
    distinct_count = len(regions)
    if distinct_count < 2 or not ({"waist", "silhouette"} & regions):
        parsed["guard_reason"] = "缺少独立的形态证据或看不清腰部和剪影"
        parsed["label"] = "unsure"
        return


def parse_packet(content, squares):
    data = review._load_strict(content)

    if not isinstance(data, dict) or set(data.keys()) != {"results"}:
        raise ValueError("packet must have exact key 'results'")
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("results must be a list")

    if not isinstance(squares, (list, tuple)):
        raise ValueError("squares must be a list or tuple")
    squares = list(squares)
    if not (1 <= len(squares) <= 5):
        raise ValueError("squares must contain 1 to 5 entries")
    for square in squares:
        review._validate_square(square)
    if len(set(squares)) != len(squares):
        raise ValueError("squares must be distinct")

    if len(results) != len(squares):
        raise ValueError("results count must match requested squares")

    parsed = []
    seen = set()
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("each result must be an object")
        square = row.get("square")
        if not isinstance(square, str):
            raise ValueError("row square must be a string")
        if square not in squares:
            raise ValueError("unexpected square %r" % square)
        if square in seen:
            raise ValueError("duplicate square %r" % square)
        seen.add(square)
        parsed.append(_parse_row(square, row))

    if seen != set(squares):
        raise ValueError("missing requested squares")

    output = {}
    for item in parsed:
        _consistency(item)
        output[item["square"]] = item

    ordered = {}
    for square in squares:
        ordered[square] = output[square]
    return ordered
