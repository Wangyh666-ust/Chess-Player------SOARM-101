"""Reuse the v5 review with enough output tokens for the five-square batch."""

import review_v5
from review_v5 import parse_packet

MAX_TOKENS = 16000


def build_packet(prompt, requests, reference):
    packet = review_v5.build_packet(prompt, requests, reference)
    packet["max_tokens"] = MAX_TOKENS
    return packet
