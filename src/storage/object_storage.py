"""
Lưu trữ file: SeaweedFS qua S3 API (mặc định) hoặc thư mục cục bộ (STORAGE_BACKEND=local).

Đổi sang AWS S3 chỉ cần đổi S3_ENDPOINT_URL và khóa truy cập.
"""

import logging
from pathlib import Path
from typing import Optional, Protocol

from src.config import settings

log = logging.getLogger(__name__)


class ObjectStorage(Protocol):
    def put(self, bucket: str, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, bucket: str, key: str) -> bytes: ...
    def delete(self, bucket: str, key: str) -> None: ...
    def presigned_url(self, bucket: str, key: str, expires_s: int = 300) -> Optional[str]: ...


class S3Storage:
    def __init__(self):
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4",
                          retries={"max_attempts": 3}),
        )
        self._ready: set = set()

    def _ensure_bucket(self, bucket: str) -> None:
        if bucket in self._ready:
            return
        from botocore.exceptions import ClientError
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError:
            self.client.create_bucket(Bucket=bucket)
            if bucket == settings.s3_bucket_uploads:
                self._apply_upload_lifecycle(bucket)
        self._ready.add(bucket)

    def _apply_upload_lifecycle(self, bucket: str) -> None:
        """Tự xóa bệnh án sau UPLOAD_TTL_DAYS ngày (nếu backend S3 hỗ trợ lifecycle)."""
        try:
            self.client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration={"Rules": [{
                    "ID": "expire-uploads", "Status": "Enabled", "Filter": {"Prefix": ""},
                    "Expiration": {"Days": settings.upload_ttl_days},
                }]},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Không đặt được lifecycle cho bucket %s: %s", bucket, exc)

    def put(self, bucket: str, key: str, data: bytes, content_type: str) -> None:
        self._ensure_bucket(bucket)
        self.client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, bucket: str, key: str) -> bytes:
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def delete(self, bucket: str, key: str) -> None:
        self.client.delete_object(Bucket=bucket, Key=key)

    def presigned_url(self, bucket: str, key: str, expires_s: int = 300) -> Optional[str]:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_s)


class LocalStorage:
    def __init__(self, root: Path = settings.local_storage_dir):
        self.root = root

    def _path(self, bucket: str, key: str) -> Path:
        path = (self.root / bucket / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError("Đường dẫn không hợp lệ")
        return path

    def put(self, bucket: str, key: str, data: bytes, content_type: str) -> None:
        path = self._path(bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, bucket: str, key: str) -> bytes:
        return self._path(bucket, key).read_bytes()

    def delete(self, bucket: str, key: str) -> None:
        self._path(bucket, key).unlink(missing_ok=True)

    def presigned_url(self, bucket: str, key: str, expires_s: int = 300) -> Optional[str]:
        return None


_storage: Optional[ObjectStorage] = None


def get_storage() -> ObjectStorage:
    global _storage
    if _storage is None:
        _storage = LocalStorage() if settings.storage_backend == "local" else S3Storage()
    return _storage
