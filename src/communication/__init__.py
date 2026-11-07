"""Giao thức giao tiếp nội bộ giữa các tác tử (MessageBus, AgentMessage, Summary)."""
from src.communication.message_bus import AgentMessage, BaseAgent, MessageBus
__all__ = ["AgentMessage", "BaseAgent", "MessageBus"]
