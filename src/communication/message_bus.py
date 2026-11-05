"""
Giao thức truyền thông Đa Tác Tử (Agent Communication & Message Bus)
Chuẩn theo Mục 5.2 Slide Developing Multiagent Systems (Intelligent System Development)

Các tác tử KHÔNG gọi hàm của nhau. Mọi tương tác đi qua MessageBus bằng AgentMessage có
performative theo tinh thần FIPA-ACL (REQUEST, INFORM, CHALLENGE, RESPONSE, FAILURE).
Bus định tuyến tin theo `receiver`, gọi `handle()` của tác tử nhận và ghi toàn bộ hội thoại
(audit trail) theo `conversation_id` — mỗi lượt chat là một conversation.
"""

import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

PERFORMATIVES = {"REQUEST", "INFORM", "CHALLENGE", "RESPONSE", "FAILURE"}

# Nơi nhận thông điệp ngay khi bus ghi nhận (để giao diện hiện "suy luận" trực tiếp). Đặt theo từng lượt chat;
# các task con tạo bằng asyncio.gather thừa hưởng context nên mọi thông điệp của lượt đó đều đi qua đây.
stream_sink: ContextVar[Optional[Callable[["AgentMessage"], None]]] = ContextVar("stream_sink", default=None)


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    performative: str
    msg_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    in_reply_to: Optional[str] = None
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))
    elapsed_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "in_reply_to": self.in_reply_to,
            "sender": self.sender,
            "receiver": self.receiver,
            "performative": self.performative,
            "msg_type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
        }


class BaseAgent:
    """Lớp cơ sở cho mọi tác tử: tự đăng ký vào bus, nhận tin qua handle(), gửi tin qua ask()."""

    name: str = "BaseAgent"
    handles: set = set()

    def __init__(self, bus: "MessageBus"):
        self.bus = bus
        bus.register(self)

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        raise NotImplementedError

    def reply(self, msg: AgentMessage, msg_type: str, payload: Dict[str, Any],
              performative: str = "INFORM") -> AgentMessage:
        return AgentMessage(
            sender=self.name, receiver=msg.sender, performative=performative, msg_type=msg_type,
            payload=payload, conversation_id=msg.conversation_id, in_reply_to=msg.message_id,
        )

    async def ask(self, receiver: str, msg_type: str, payload: Dict[str, Any], conversation_id: str,
                  performative: str = "REQUEST") -> AgentMessage:
        return await self.bus.send(AgentMessage(
            sender=self.name, receiver=receiver, performative=performative, msg_type=msg_type,
            payload=payload, conversation_id=conversation_id,
        ))


class MessageBus:
    """Kênh truyền thông trung tâm: định tuyến tin nhắn và lưu audit trail theo từng conversation."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.transcripts: Dict[str, List[AgentMessage]] = {}

    def register(self, agent: BaseAgent) -> None:
        self.agents[agent.name] = agent

    def _record(self, msg: AgentMessage) -> None:
        self.transcripts.setdefault(msg.conversation_id, []).append(msg)
        sink = stream_sink.get()
        if sink is not None:
            try:
                sink(msg)
            except Exception:  # noqa: BLE001 — lỗi ở phía hiển thị không được làm hỏng lượt hội chẩn
                log.exception("stream_sink lỗi")

    async def send(self, msg: AgentMessage) -> AgentMessage:
        if msg.performative not in PERFORMATIVES:
            raise ValueError(f"Performative không hợp lệ: {msg.performative}")
        self._record(msg)
        agent = self.agents.get(msg.receiver)
        started = time.perf_counter()
        if agent is None:
            reply = AgentMessage(sender="MessageBus", receiver=msg.sender, performative="FAILURE",
                                 msg_type="UNKNOWN_RECEIVER", payload={"error": msg.receiver},
                                 conversation_id=msg.conversation_id, in_reply_to=msg.message_id)
        elif msg.msg_type not in agent.handles:
            reply = agent.reply(msg, "NOT_UNDERSTOOD", {"error": f"{agent.name} không xử lý {msg.msg_type}"},
                                performative="FAILURE")
        else:
            try:
                reply = await agent.handle(msg)
            except Exception as exc:  # noqa: BLE001 — lỗi của một tác tử không làm sập cả hội đồng
                log.exception("Tác tử %s lỗi khi xử lý %s", agent.name, msg.msg_type)
                reply = agent.reply(msg, f"{msg.msg_type}_FAILED", {"error": f"{type(exc).__name__}: {exc}"},
                                    performative="FAILURE")
        reply.elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._record(reply)
        return reply

    def pop_transcript(self, conversation_id: str) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.transcripts.pop(conversation_id, [])]
