"""
alterEgo - Callback de observabilidad para LangGraph.
Registra cada paso del agente: herramientas, LLM, tiempos, errores.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class AlterEgoCallback(BaseCallbackHandler):
    """Callback que registra toda la actividad del agente en tiempo real."""

    def __init__(self, log_dir: Path | None = None, verbose: bool = True):
        self.log_dir = log_dir or Path("harness/reports")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Identificador de la sesión
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trace_{self.session_id}.jsonl"

        # Estado interno
        self.events: list[dict[str, Any]] = []
        self._step_start: float = 0.0
        self._tool_name: str = ""

    def _log(self, event: str, data: dict[str, Any] | None = None):
        """Registra un evento en memoria y en disco."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {},
        }
        self.events.append(entry)

        # Escribir en archivo JSONL (un JSON por línea)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        # Mostrar en consola si verbose
        if self.verbose:
            self._print_event(event, data or {})

    def _print_event(self, event: str, data: dict[str, Any]):
        """Imprime el evento formateado en consola."""
        if event == "tool_start":
            tool = data.get("tool", "?")
            inp = data.get("input", "")[:100]
            print(f"    🔧 {tool}({inp})")

        elif event == "tool_end":
            duration = data.get("duration_ms", 0)
            print(f"    ✅ resultado ({duration}ms)")

        elif event == "llm_start":
            print("    🤖 Pensando...")

        elif event == "llm_end":
            duration = data.get("duration_ms", 0)
            print(f"    💬 Respuesta generada ({duration}ms)")

        elif event == "error":
            error = data.get("error", "?")
            print(f"    ❌ Error: {error}")

    # ── Eventos de herramientas ──

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._step_start = time.time()
        self._tool_name = serialized.get("name", "unknown")
        self._log("tool_start", {
            "tool": self._tool_name,
            "input": input_str[:200],
        })

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        duration = round((time.time() - self._step_start) * 1000, 2)
        self._log("tool_end", {
            "tool": self._tool_name,
            "duration_ms": duration,
            "output_preview": str(output)[:200],
        })

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._log("tool_error", {
            "tool": self._tool_name,
            "error": str(error),
        })

    # ── Eventos del LLM ──

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._step_start = time.time()
        self._log("llm_start", {
            "model": serialized.get("name", "?"),
            "prompts_count": len(prompts),
        })

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        duration = round((time.time() - self._step_start) * 1000, 2)

        # Intentar extraer info de tokens
        token_usage = {}
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        self._log("llm_end", {
            "duration_ms": duration,
            "token_usage": token_usage,
        })

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._log("llm_error", {"error": str(error)})

    # ── Eventos de cadena/agente ──

        # ── Eventos de cadena/agente ──

    def on_chain_start(
        self, serialized: dict[str, Any] | None, inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        if serialized is None:
            self._log("chain_start", {"chain": "internal"})
            return
        chain_name = str(serialized.get("name", serialized.get("id", ["?"])[-1]))
        self._log("chain_start", {"chain": chain_name})

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        # outputs puede ser dict, list, AIMessage, str, etc.
        if isinstance(outputs, dict):
            self._log("chain_end", {"output_keys": list(outputs.keys())})
        elif isinstance(outputs, list):
            self._log("chain_end", {"output_count": len(outputs)})
        elif isinstance(outputs, str):
            self._log("chain_end", {"output_preview": outputs[:100]})
        else:
            self._log("chain_end", {"output_type": type(outputs).__name__})

    # ── Resumen y utilidades ──

    def summary(self) -> dict[str, Any]:
        """Devuelve un resumen de la sesión."""
        tool_calls = [e for e in self.events if e["event"] == "tool_start"]
        llm_calls = [e for e in self.events if e["event"] == "llm_start"]
        errors = [e for e in self.events if "error" in e["event"]]

        tool_names = [e["data"].get("tool", "?") for e in tool_calls]
        total_tool_time = sum(
            e["data"].get("duration_ms", 0)
            for e in self.events
            if e["event"] == "tool_end"
        )
        total_llm_time = sum(
            e["data"].get("duration_ms", 0)
            for e in self.events
            if e["event"] == "llm_end"
        )

        return {
            "session_id": self.session_id,
            "total_events": len(self.events),
            "tool_calls": len(tool_calls),
            "tool_names": tool_names,
            "llm_calls": len(llm_calls),
            "errors": len(errors),
            "total_tool_time_ms": round(total_tool_time, 2),
            "total_llm_time_ms": round(total_llm_time, 2),
            "log_file": str(self.log_file),
        }

    def print_summary(self):
        """Imprime el resumen en consola."""
        s = self.summary()
        print(f"\n{'─' * 50}")
        print(f"  📊 Resumen de la sesión: {s['session_id']}")
        print(f"{'─' * 50}")
        print(f"  Herramientas llamadas: {s['tool_calls']}")
        if s["tool_names"]:
            print(f"    → {', '.join(s['tool_names'])}")
        print(f"  Llamadas al LLM: {s['llm_calls']}")
        print(f"  Tiempo en herramientas: {s['total_tool_time_ms']}ms")
        print(f"  Tiempo en LLM: {s['total_llm_time_ms']}ms")
        print(f"  Errores: {s['errors']}")
        print(f"  Log completo: {s['log_file']}")
        print(f"{'─' * 50}\n")
