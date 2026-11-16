"""Place the checked field reference before the target crops."""

from review_v5_long import MAX_TOKENS, build_packet as _build_packet, parse_packet

__all__ = ["build_packet", "parse_packet", "MAX_TOKENS"]


def build_packet(prompt, requests, reference):
    packet = _build_packet(prompt, requests, reference)
    message = packet["messages"][0]
    parts = message["content"]
    introduction = {
        "type": "text",
        "text": ("下图上方是同款棋子近照，下方是 60 张实机 C1/C2 裁图。"
                 "编号只对应参考样本。先看这张参考图，再看随后的现场目标。"),
    }
    message["content"] = [parts[0], introduction, parts[-1], *parts[1:-2]]
    return packet
