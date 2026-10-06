from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.llm.provider import get_llm
from src.memory.vector_memory import VectorMemory
from src.tools.get_profile import create_profile_tool
from src.tools.search_by_date import create_date_search_tool
from src.tools.search_memory import create_search_tool


class AlterEgoAgent:
    """Agente principal de alterEgo."""

    def __init__(self, config: dict):
        self.config = config
        self.llm = get_llm(config)
        self.memory = VectorMemory(config)
        self.profile = self._load_profile()
        self.agent: Any = None  # ← Any, sin importar tipo específico

    def setup(self) -> None:
        """Inicializa el agente con herramientas y prompt."""
        if not self.memory.load():
            raise RuntimeError(
                "No hay índice vectorial. Ejecuta: python main.py --build-index"
            )

        tools = [
            create_search_tool(self.memory),
            create_profile_tool(self.profile),
            create_date_search_tool(self.memory),
        ]

        identity_prompt = self._load_identity_prompt()

        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=identity_prompt,
        )

        print("  ✅ Agente alterEgo inicializado (LangGraph).")

    def _get_agent(self) -> Any:
        """Devuelve el agente, asegurando que está inicializado."""
        if self.agent is None:
            self.setup()
        assert self.agent is not None
        return self.agent

    def chat(
        self,
        user_input: str,
        chat_history: list | None = None,
        callbacks: list | None = None,
    ) -> str:
        """Envía un mensaje al agente y devuelve la respuesta."""
        agent = self._get_agent()

        messages: list[Any] = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_input))

        # Config con callbacks
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks

        result = agent.invoke({"messages": messages}, config=config)

        ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
        return ai_messages[-1].content if ai_messages else ""

    def generate_biography(self, callbacks: list | None = None) -> str:
        """Genera la biografía usando el agente con múltiples búsquedas."""
        agent = self._get_agent()

        prompt = (
            "Quiero que escribas mi biografía completa. Para ello:\n"
            "1. Primero usa get_profile para saber quién soy.\n"
            "2. Luego usa search_memory con estas consultas (una por una):\n"
            "   - 'infancia, familia, pueblo, casa'\n"
            "   - 'trabajo, profesión, carrera, informática'\n"
            "   - 'aficiones, ciclismo, bicicleta'\n"
            "   - 'Beatles, música'\n"
            "   - 'física, ciencia, religión'\n"
            "   - 'viajes, lugares, vacaciones'\n"
            "3. Con toda la información recopilada, escribe mi biografía "
            "en primera persona. Sé fiel a los datos. No inventes. "
            "Si no hay datos suficientes para un aspecto, omítelo o di "
            "que no lo recuerdas con claridad.\n"
            "4. Extensión: entre 600 y 1200 palabras. Formato Markdown."
        )

        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks

        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
        return ai_messages[-1].content if ai_messages else ""

    def _load_profile(self) -> dict:
        profile_path = Path("config/user_profile.yaml")
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_identity_prompt(self) -> str:
        prompt_path = Path("prompts/identity.md")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        name = self.profile.get("name", "alterEgo")
        return (
            f"Eres {name}. Respondes como si fueras esta persona. "
            f"Usas tus herramientas para buscar información real. "
            f"Nunca inventas datos."
        )
