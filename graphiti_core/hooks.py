"""
Graphiti Hooks - 写入前后钩子

提供在数据写入前后执行自定义逻辑的能力，用于：
1. 公理约束验证（基数、值域）
2. 数据过滤和转换
3. 审计日志记录
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode


class PreWriteHook(ABC):
    """写入前钩子基类

    在数据写入图数据库之前调用，可用于：
    - 验证公理约束
    - 过滤不符合条件的边
    - 抛出异常阻止写入

    使用示例:
        class AxiomValidationHook(PreWriteHook):
            async def validate_edges(self, edges, nodes, group_id):
                for edge in edges:
                    if violates_cardinality(edge):
                        raise CardinalityViolationError(...)
                return edges

        graphiti = Graphiti(..., pre_write_hook=AxiomValidationHook())
    """

    @abstractmethod
    async def validate_edges(
        self,
        edges: list['EntityEdge'],
        nodes: list['EntityNode'],
        group_id: str,
    ) -> list['EntityEdge']:
        """
        验证并处理即将写入的边

        Args:
            edges: 即将写入的边列表
            nodes: 关联的节点列表
            group_id: 租户 ID

        Returns:
            处理后的边列表（可过滤或修改）

        Raises:
            Exception: 如果验证失败，抛出异常阻止写入
        """
        pass
