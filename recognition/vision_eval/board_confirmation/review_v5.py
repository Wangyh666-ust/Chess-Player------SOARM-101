import review_v4

MAX_TOKENS = 8192

def parse_packet(content, squares):
    results = review_v4.parse_packet(content, squares)
    for record in results.values():
        if record["label"] in ("wP", "bP") and not any(
            view["base"] == "target" and view["top"] == "sphere"
            for view in record["observations"].values()
        ):
            record["label"] = "unsure"
            record["guard_reason"] = "没有一路在目标棋子上报告可辨认的完整球面"
    return results


def build_packet(prompt, requests, reference):
    packet = review_v4.build_packet(prompt, requests, reference)
    packet["thinking"] = {"type": "enabled"}
    packet["reasoning_effort"] = "high"
    packet["max_tokens"] = MAX_TOKENS
    packet.pop("temperature", None)
    return packet
