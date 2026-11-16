import hashlib
import json
from pathlib import Path

from multimodel_review_trial import (
    _run_dashscope,
    build_messages,
    load_inputs,
)


def replace_prompt(messages: list, prompt: str) -> list:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if not messages:
        raise ValueError("messages must not be empty")
    first = messages[0]
    content = first.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("first message must have a non-empty content list")
    block = content[0]
    if not isinstance(block, dict) or block.get("type") != "text":
        raise ValueError("first content block must be a text block")
    new_block = dict(block)
    new_block["text"] = prompt
    new_content = list(content)
    new_content[0] = new_block
    new_first = dict(first)
    new_first["content"] = new_content
    new_messages = list(messages)
    new_messages[0] = new_first
    return new_messages


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=False)

    views, _prompt, reference, reference_sha = load_inputs()

    prompt_path = Path(__file__).resolve().with_name("prompt_low_rook_v8.txt")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    messages = build_messages(views, reference)
    messages = replace_prompt(messages, prompt)

    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    image_blocks = 0
    for message in messages:
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_blocks += 1

    result = _run_dashscope("max", messages, list(views))

    result["image_blocks"] = image_blocks
    result["reference_sha256"] = reference_sha
    result["prompt_sha256"] = prompt_sha

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
