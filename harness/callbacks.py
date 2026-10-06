"""
alterEgo - Callbacks de observabilidad para LangChain.
Registra cada paso del agente para análisis posterior.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class AlterEgoCallback(BaseCallbackHandler):
    """Callback que registra toda la actividad del agente."""

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir or Path("harness/reports")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trace_{self.run_id}.jsonl"
        self.events: list[dict] = []
        self._step_start: float = 0.0

    def _log(self, event: str, data: dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data,
        }
        self.events.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # ── Eventos del agente ──

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        self._step_start = time.time()
        self._log("llm_start", {"prompts_count": len(prompts)})

    def on_llm_end(self, response: Any, **kwargs):
        duration = round((time.time() - self._step_start) * 1000, 2)
        self._log("llm_end", {"duration_ms": duration})

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self._step_start = time.time()
        tool_name = serialized.get("name", "unknown")
        self._log("tool_start", {"tool": tool_name, "input": input_str[:200]})

    def on_tool_end(self, output: str, **kwargs):
        duration = round((time.time() - self._step_start) * 1000, 2)
        self._log("tool_end", {"duration_ms": duration, "output_preview": str(output)[:200]})

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        self._log("chain_start", {"chain": serialized.get("name", "?")})

    def on_chain_end(self, outputs: dict, **kwargs):
        self._log("chain_end", {"keys": list(outputs.keys())})

    def on_agent_action(self, action: Any, **kwargs):
        self._log("agent_action", {
            "tool": getattr(action, "tool", "?"),
            "input": str(getattr(action, "tool_input", ""))[:200],
        })

    def on_agent_finish(self, finish: Any, **kwargs):
        self._log("agent_finish", {"output": str(getattr(finish, "return_values", {}))[:300]})

    # ── Resumen ──

    def summary(self) -> dict:
        tools_used = [e for e in self.events if e["event"] == "tool_start"]
        return {
            "run_id": self.run_id,
            "total_events": len(self.events),
            "tools_called": len(tools_used),
            "tool_names": [e["data"].get("tool", "?") for e in tools_used],
            "log_file": str(self.log_file),
        }
