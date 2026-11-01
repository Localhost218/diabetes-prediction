import os
import sys
from pathlib import Path

# Test chạy ở chế độ không LLM để kết quả ổn định, không tốn API
os.environ["LLM_ENABLED"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
