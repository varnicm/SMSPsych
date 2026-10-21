#!/usr/bin/env python
"""
annotate_grok.py - candidate PT/PF labels from xAI Grok.

xAI exposes an OpenAI-compatible API, so we reuse the openai client
pointed at the xAI base URL.

Setup:
    pip install openai
    export XAI_API_KEY="YOUR_API_KEY"
Run:
    python3 annotate_grok.py
"""

import os
from openai import OpenAI
from common import run_provider

# pin an exact model id for reproducibility
MODEL = "grok-4.3"

client = OpenAI(
    api_key=os.environ["XAI_API_KEY"],
    base_url="https://api.x.ai/v1",
)


def call_model(system, user):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    run_provider(f"grok_{MODEL}", call_model)
