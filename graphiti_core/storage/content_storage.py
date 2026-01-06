"""
Content Storage Abstraction Layer

提供统一的内容存储接口，支持多种存储后端（Redis、本地文件、OSS、S3等）
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class StorageResult:
    """存储结果"""

    file_path: str | None  # 文件路径（REDIS 模式为 None）
    content_hash: str  # SHA-256
    file_size: int  # 字节数


class ContentStorage(Protocol):
    """内容存储接口 - 所有存储类型必须实现此协议"""

    async def store(
        self,
        episode_uuid: str,
        group_id: str,
        content: str,
    ) -> StorageResult:
        """
        存储内容，返回存储结果

        Args:
            episode_uuid: Episode UUID
            group_id: 租户 ID（用于路径组织）
            content: 要存储的内容

        Returns:
            StorageResult: 包含 file_path, content_hash, file_size
        """
        ...

    async def retrieve(
        self,
        file_path: str | None,
        content: str | None,
        max_length: int | None = None,
    ) -> str:
        """
        读取内容

        Args:
            file_path: 文件路径（LOCAL/OSS/S3）或 None（REDIS）
            content: Redis 模式下的完整内容（兼容读取）
            max_length: 截断长度（可选）

        Returns:
            内容字符串
        """
        ...

    async def compute_hash(self, content: str) -> str:
        """
        计算 SHA-256 hash

        Args:
            content: 要计算 hash 的内容

        Returns:
            SHA-256 hash 字符串
        """
        ...
