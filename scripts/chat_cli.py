"""
Trò chuyện với hội đồng đa tác tử ngay trên terminal (không cần giao diện web).

    python scripts/chat_cli.py
    python scripts/chat_cli.py --trace      # in thêm luồng thông điệp giữa các tác tử
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import OrchestratorAgent  # noqa: E402
from src.schemas.patient_profile import PatientProfile  # noqa: E402


async def main(trace: bool) -> None:
    orchestrator, profile = OrchestratorAgent(), PatientProfile()
    print("Diabetes Prediction — gõ tin nhắn, 'reset' để xóa hồ sơ, 'exit' để thoát.\n")
    while True:
        try:
            text = input("Bạn: ").strip()
        except EOFError:
            break
        if text in ("exit", "quit"):
            break
        if text == "reset":
            profile = PatientProfile()
            continue
        if not text:
            continue
        result = await orchestrator.run_turn(profile, text, [])
        if trace:
            print("  kế hoạch:", " → ".join(result["plan"]))
            for m in result["transcript"]:
                print(f"  {m['sender']:>20} → {m['receiver']:<20} {m['performative']:<9} {m['msg_type']}")
        print(f"\nDiabetes Prediction:\n{result['text']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    asyncio.run(main(parser.parse_args().trace))
