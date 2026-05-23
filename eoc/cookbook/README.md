# EOC Cookbook

A HuggingFace Cookbook-style recipe demonstrating the EOC four-stage cascade (cache, KV, graph, neural) with HuggingFace AI Energy Score integration. Each resolved query emits a content-addressed receipt carrying its joule cost, and the recipe aggregates joules per correct answer across a small evaluation set. Production deployments swap in their own knowledge graph and neural backend; the cascade contract — *first sufficient stage wins, joules accounted at every step* — is the substrate.

- **Narrative**: [eoc-cascade-with-hf-energy-score.md](./eoc-cascade-with-hf-energy-score.md) — the recipe walkthrough.
- **Runnable code**: [eoc-cascade.py](./eoc-cascade.py) — end-to-end implementation with a phi-3-mini fallback.
- **Dependencies**: [requirements.txt](./requirements.txt) — pinned versions.
- **SCI-for-AI mapping** (companion submission): [../standards/sci-for-ai-submission.md](../standards/sci-for-ai-submission.md).

Licensed CC-BY-4.0 (text) and Apache-2.0 (code), consistent with the parent `eoc/` directory.
