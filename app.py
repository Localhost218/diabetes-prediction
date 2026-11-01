"""
Diabetes Prediction — Hệ thống Chatbot Đa Tác Tử sàng lọc nguy cơ đái tháo đường.

    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Giao diện chatbot: http://localhost:8000/   ·   Swagger: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api import routes_chat, routes_session, routes_system
from src.api.deps import get_orchestrator
from src.config import settings
from src.storage.db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    get_orchestrator()  # nạp mô hình ML, embedding và đăng ký các tác tử ngay khi khởi động
    yield


app = FastAPI(
    title="Diabetes Prediction — Multi-Agent Diabetes Screening",
    description="Chatbot đa tác tử: Orchestrator, Conversation, Document, Prediction, Knowledge, "
                "ConsensusCritic, TreatmentRL, Response.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "tauri://localhost"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(routes_system.router)
app.include_router(routes_session.router)
app.include_router(routes_chat.router)


@app.get("/chat", include_in_schema=False)
def chat_redirect():
    return RedirectResponse("/")


# Giao diện Diabetes Prediction (bản build của frontend/) — mount sau cùng để không che các route /api
if settings.frontend_dist.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
