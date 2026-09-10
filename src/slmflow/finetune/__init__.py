"""Phase 8: LoRA adapter loading. Training happens off-machine.

Decoupled by design: the orchestration core must not import anything from here
(``BUILD_BRIEF.md`` Phase 8), so nothing in ``slmflow/`` outside this package
references it.
"""

from slmflow.finetune.adapters import AdapterSpec, attach_adapter, list_adapters

__all__ = ["AdapterSpec", "attach_adapter", "list_adapters"]
