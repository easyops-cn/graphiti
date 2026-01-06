"""
Redis Content Storage

Redis 存储实现 - 直接存储在 FalkorDB 节点中（兼容模式）
此模式下，content 完整存储在 EpisodicNode.content 字段中
"""

import hashlib

from .content_storage import ContentStorage, StorageResult


class RedisContentStorage:
    """Redis 存储实现 - 向后兼容模式"""

    async def store(
        self,
        episode_uuid: str,
        group_id: str,
        content: str,
    ) -> StorageResult:
        """
        Redis 模式：不实际存储文件，返回元数据供 FalkorDB 使用

        Args:
            episode_uuid: Episode UUID
            group_id: 租户 ID
            content: 要存储的内容

        Returns:
            StorageResult: file_path 为 None，content_hash 和 file_size 计算值
        """
        content_hash = await self.compute_hash(content)
        return StorageResult(
            file_path=None,  # 无文件路径
            content_hash=content_hash,
            file_size=len(content.encode("utf-8")),
        )

    async def retrieve(
        self,
        file_path: str | None,
        content: str | None,
        max_length: int | None = None,
    ) -> str:
        """
        Redis 模式：content 已经在 EpisodicNode.content 中，直接返回

        Args:
            file_path: 文件路径（Redis 模式下为 None）
            content: 完整内容（从 FalkorDB 读取）
            max_length: 截断长度（可选）

        Returns:
            内容字符串（可能截断）
        """
        if content is None:
            return ""

        if max_length and len(content) > max_length:
            return content[:max_length]

        return content

    async def compute_hash(self, content: str) -> str:
        """
        计算 SHA-256 hash

        Args:
            content: 要计算 hash 的内容

        Returns:
            SHA-256 hash 字符串（十六进制）
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
