"""
Local File Storage

本地文件系统存储实现
将 Episode 内容存储为本地文件（.txt 格式，纯文本，无 JSON 转义）
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from .content_storage import ContentStorage, StorageResult


class LocalFileStorage:
    """本地文件系统存储"""

    def __init__(self, base_path: str = "./data/episodes"):
        """
        初始化本地文件存储

        Args:
            base_path: 基础目录路径，默认为 ./data/episodes
        """
        self.base_path = Path(base_path)

    async def store(
        self,
        episode_uuid: str,
        group_id: str,
        content: str,
    ) -> StorageResult:
        """
        存储内容到本地文件

        文件路径格式: {base_path}/{group_id}/{year}/{month}/{content_hash}.txt
        使用 content_hash 作为文件名，相同内容自动去重

        Args:
            episode_uuid: Episode UUID（仅用于日志，不影响文件名）
            group_id: 租户 ID
            content: 要存储的内容

        Returns:
            StorageResult: 包含文件路径、hash、大小
        """
        # 1. 计算 content hash
        content_hash = await self.compute_hash(content)

        # 2. 生成文件路径（使用 hash 作为文件名）
        now = datetime.now(timezone.utc)
        file_path = f"{group_id}/{now.year}/{now.month:02d}/{content_hash}.txt"
        full_path = self.base_path / file_path

        # 3. 如果文件已存在，直接返回（去重）
        if full_path.exists():
            existing_size = full_path.stat().st_size
            return StorageResult(
                file_path=file_path,
                content_hash=content_hash,
                file_size=existing_size,
            )

        # 4. 创建目录
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. 写入新文件（纯文本格式，UTF-8 编码）
        content_bytes = content.encode("utf-8")
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(content_bytes)

        # 6. 返回存储结果
        return StorageResult(
            file_path=file_path,
            content_hash=content_hash,
            file_size=len(content_bytes),
        )

    async def retrieve(
        self,
        file_path: str | None,
        content: str | None,
        max_length: int | None = None,
    ) -> str:
        """
        从本地文件读取内容

        Args:
            file_path: 文件相对路径
            content: 兼容 Redis 模式的完整内容（如果 file_path 为 None）
            max_length: 截断长度（可选）

        Returns:
            内容字符串（可能截断）

        Raises:
            FileNotFoundError: 文件不存在
        """
        # 兼容 Redis 模式
        if file_path is None:
            return content or ""

        full_path = self.base_path / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {file_path}")

        # 读取文件
        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            content_str = await f.read()

        # 截断（如果需要）
        if max_length and len(content_str) > max_length:
            return content_str[:max_length]

        return content_str

    async def compute_hash(self, content: str) -> str:
        """
        计算 SHA-256 hash

        Args:
            content: 要计算 hash 的内容

        Returns:
            SHA-256 hash 字符串（十六进制）
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
