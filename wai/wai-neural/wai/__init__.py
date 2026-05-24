"""WAI v0.4 — generative semantic communication: transport + execution.

Send the understanding, not the content. The sink's own conforming
generative model regenerates semantically-equivalent media. The model is
a requirement of existence, never transported, never hash-pinned.
"""
from .container import Wai, VERBS
from .encode import encode_replicate, encode_create, encode_improve
from .runtime import WaiRuntime, InertWaiError, Result
__all__ = ["Wai", "VERBS", "encode_replicate", "encode_create",
           "encode_improve", "WaiRuntime", "InertWaiError", "Result"]
