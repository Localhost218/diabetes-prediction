FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# PyTorch bản CPU (nhẹ hơn nhiều so với bản CUDA mặc định)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
# Tải sẵn mô hình embedding vào image để khởi động không cần mạng
RUN python -c "from src.knowledge.embedder import get_embedder; get_embedder()"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
