"""Fixed probe set builder."""
from __future__ import annotations

from typing import List
from datasets import Dataset
from curvature_dpo.types import ProbeItem
from curvature_dpo.data.ultrafeedback import as_text, response_text


def build_probe_set(ds: Dataset, tokenizer=None) -> List[ProbeItem]:
    """
    Converts a dataset shard into a list of ProbeItem objects.
    Extracts 'prompt' and 'chosen' as the base for curvature estimation.
    """
    probes = []
    for item in ds:
        probe = ProbeItem(
            prompt=as_text(item["prompt"]),
            response=response_text(item["chosen"]),
        )
        if tokenizer:
            probe.prompt_ids = tokenizer(probe.prompt, add_special_tokens=True)["input_ids"]
            probe.response_ids = tokenizer(probe.response, add_special_tokens=False)["input_ids"]
            
        probes.append(probe)
    return probes
