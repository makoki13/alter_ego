"""
alterEgo - Agente principal usando LangChain.
Combina LLM + herramientas + memoria + prompts.
"""

from pathlib import Path
from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        self.agent_executor: AgentExecutor | None = None

    def setup(self):
        """Inicializa el agente con herramientas y prompt."""
        # Cargar memoria vectorial
        if not self.memory.load():
            raise RuntimeError(
                "No hay índice vectorial. Ejecuta: python main.py --build-index"
            )

        # Crear herramientas
        tools = [
            create_search_tool(self.memory),
            create_profile_tool(self.profile),
            create_date_search_tool(self.memory),
        ]

        # Cargar prompt de identidad
        identity_prompt = self._load_identity_prompt()

        # Prompt del agente
        prompt = ChatPromptTemplate.from_messages([
            ("system", identity_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Crear agente
        agent = create_openai_tools_agent(self.llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        print("  ✅ Agente alterEgo inicializado.")

    def chat(self, user_input: str, chat_history: list | None = None) -> str:
        """Envía un mensaje al agente y devuelve la respuesta."""
        if self.agent_executor is None:
            self.setup()

        result = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history or [],
        })

        return result.get("output", "")

    def generate_biography(self) -> str:
        """Genera la biografía usando el agente."""
        if self.agent_executor is None:
            self.setup()

        prompt = (
            "Usa todas tus herramientas para recopilar información sobre mí: "
            "mi perfil, mis recuerdos de infancia, mi trabajo, mis aficiones, "
            "mis relaciones, mis viajes. Con toda esa información, escribe "
            "mi biografía completa en primera persona. Sé fiel a los datos."
        )

        result = self.agent_executor.invoke({
            "input": prompt,
            "chat_history": [],
        })

        return result.get("output", "")

    def _load_profile(self) -> dict:
        import yaml
        profile_path = Path("config/user_profile.yaml")
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_identity_prompt(self) -> str:
        prompt_path = Path("prompts/identity.md")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        # Fallback
        name = self.profile.get("name", "alterEgo")
        return (
            f"Eres {name}. Respondes como si fueras esta persona. "
            f"Usas tus herramientas para buscar información real antes de responder. "
            f"Nunca inventas datos. Si no sabes algo, lo dices."
        )
