#!/usr/bin/env python
"""
annotate_openai.py - candidate PT/PF labels from OpenAI GPT.

Setup:
    pip install openai
    export OPENAI_API_KEY="YOUR_API_KEY"
Run:
    python annotate_openai.py
"""

import os
from openai import OpenAI
from common import run_provider

# pin an exact model id for reproducibility (change to your available model)
MODEL = "gpt-5.5-2026-04-23"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_model(system, user):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=1.0,
        #max_tokens=1024,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    run_provider(f"openai_{MODEL}", call_model)
