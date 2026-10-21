#!/usr/bin/env python
"""
annotate_anthropic.py - candidate PT/PF labels from Anthropic Claude.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="YOUR_API_KEY"
Run:
    python3 annotate_anthropic.py
"""

import os
import anthropic
from common import run_provider
#export ANTHROPIC_API_KEY="YOUR_API_KEY"
# pin an exact model id for reproducibility
MODEL = "claude-opus-4-8"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def call_model(system, user):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=1.0,
        system=system,                       # Claude takes system separately
        messages=[{"role": "user", "content": user}],
    )
    # response content is a list of blocks; concatenate text blocks
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    run_provider(f"anthropic_{MODEL}", call_model)
