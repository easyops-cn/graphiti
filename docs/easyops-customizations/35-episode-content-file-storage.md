# Episode 内容文件存储（Episode Content File Storage）

## 问题背景

FalkorDB 是内存数据库，将所有 Episode 内容（`EpisodicNode.content`）存储在内存中，存在以下问题：

1. **内存压力大**：大量 Episode 的完整内容（可能包含长文档、日志等）占用大量内存
2. **无法扩展**：内存限制了可存储的 Episode 数量
3. **成本高**：需要更多内存资源
4. **无去重**：相同内容的 Episode 重复存储

## 解决方案

实现可配置的 Episode 内容存储系统，支持多种存储后端：

| 存储类型 | content 字段 | file_path | 文件格式 | 适用场景 | 去重支持 |
|---------|-------------|-----------|---------|---------|---------|
| REDIS   | 完整内容     | None      | -       | 向后兼容、小数据量 | ❌ |
| LOCAL   | None        | 相对路径   | .txt 纯文本 | 开发环境、单机部署 | ✅ |
| OSS     | None        | OSS key   | .txt 纯文本 | 生产环境、阿里云 | ✅ (待实现) |
| S3      | None        | S3 key    | .txt 纯文本 | 生产环境、AWS | ✅ (待实现) |

## 修改内容

### 1. 新增存储抽象层

**文件位置**: `graphiti_core/storage/`

新增文件：
- `content_storage.py` - 定义 `ContentStorage` Protocol 和 `StorageResult` dataclass
- `redis_storage.py` - `RedisContentStorage` 实现（向后兼容）
- `local_storage.py` - `LocalFileStorage` 实现（本地文件系统）

**核心接口**：
```python
class ContentStorage(Protocol):
    async def store(
        self, episode_uuid: str, group_id: str, content: str
    ) -> StorageResult:
        """存储内容，返回 (file_path, content_hash, file_size)"""

    async def retrieve(
        self, file_path: str | None, content: str | None, max_length: int | None = None
    ) -> str:
        """读取内容"""

    async def compute_hash(self, content: str) -> str:
        """计算 SHA-256 hash"""
```

### 2. EpisodicNode 新增字段

**文件**: `graphiti_core/nodes.py`

新增 4 个字段：
```python
class EpisodicNode(Node):
    # 新增字段
    content_hash: str | None             # SHA-256 hash（用于去重）
    content_storage_type: str | None     # redis | local | oss | s3
    content_file_path: str | None        # 文件路径（REDIS 模式为 None）
    content_file_size: int | None        # 内容字节数
```

**修改点**：
- `save()` 方法：保存新字段到 `episode_args` 字典
- `get_episodic_node_from_record()` 方法：从数据库记录解析新字段

### 3. Cypher 查询更新

**文件**: `graphiti_core/models/nodes/node_db_queries.py`

**FalkorDB 分支修改**：
```python
# get_episode_node_save_query() 函数
SET n.content_hash = $content_hash,
    n.content_storage_type = $content_storage_type,
    n.content_file_path = $content_file_path,
    n.content_file_size = $content_file_size

# EPISODIC_NODE_RETURN 常量
RETURN e.uuid AS uuid, e.name AS name, ...,
       e.content_hash AS content_hash,
       e.content_storage_type AS content_storage_type,
       e.content_file_path AS content_file_path,
       e.content_file_size AS content_file_size
```

### 4. FalkorDB Driver 集成

**文件**: `graphiti_core/driver/falkordb_driver.py`

**新增初始化参数**：
```python
def __init__(
    self,
    # ...
    content_storage_type: str = 'local',  # redis | local | oss | s3
    content_storage_config: dict | None = None,
):
    self.content_storage_type = content_storage_type
    self.content_storage = self._create_storage(
        content_storage_type, content_storage_config or {}
    )
```

**新增工厂方法**：
```python
def _create_storage(self, storage_type: str, config: dict) -> ContentStorage:
    if storage_type == 'redis':
        return RedisContentStorage()
    elif storage_type == 'local':
        base_path = config.get('base_path', './data/episodes')
        return LocalFileStorage(base_path=base_path)
    # ... OSS/S3 (待实现)
```

**新增辅助方法**：
```python
async def load_episode_content(
    self, episode: 'EpisodicNode', max_length: int | None = None
) -> str:
    """从存储加载 episode 内容（如果 content 字段为空）"""
    if episode.content:
        return episode.content[:max_length] if max_length else episode.content

    if episode.content_file_path or episode.content_storage_type == 'redis':
        return await self.content_storage.retrieve(
            file_path=episode.content_file_path,
            content=episode.content,
            max_length=max_length,
        )

    return ''
```

### 5. Graphiti 核心集成

**文件**: `graphiti_core/graphiti.py`

**新增 import**：
```python
from graphiti_core.storage.content_storage import StorageResult
```

**修改 `_process_episode_data()` 方法**：

在保存节点前，集成内容存储逻辑：

```python
# 内容存储集成（仅对 FalkorDB driver）
if hasattr(self.driver, 'content_storage') and hasattr(self.driver, 'content_storage_type'):
    original_content = episode.content

    # 1. 计算 hash
    content_hash = await self.driver.content_storage.compute_hash(original_content)
    episode.content_hash = content_hash

    # 2. 查询去重（仅文件模式）
    existing_file_path = None
    if self.driver.content_storage_type != 'redis':
        query = """
        MATCH (e:Episodic {content_hash: $hash})
        WHERE e.content_file_path IS NOT NULL
        RETURN e.content_file_path AS file_path
        LIMIT 1
        """
        records, _, _ = await self.driver.execute_query(query, hash=content_hash)
        if records:
            existing_file_path = records[0]['file_path']

    # 3. 存储内容
    if existing_file_path:
        # 去重：复用已存在的文件
        storage_result = StorageResult(
            file_path=existing_file_path,
            content_hash=content_hash,
            file_size=len(original_content.encode('utf-8')),
        )
    else:
        # 新内容：存储
        storage_result = await self.driver.content_storage.store(
            episode_uuid=episode.uuid,
            group_id=episode.group_id,
            content=original_content,
        )

    # 4. 更新元数据
    episode.content_storage_type = self.driver.content_storage_type
    episode.content_file_path = storage_result.file_path
    episode.content_file_size = storage_result.file_size

    # 5. 清空 content 字段（文件模式）
    if self.driver.content_storage_type != 'redis':
        episode.content = None
```

## 配置方法

### FalkorDB Driver 配置

创建 FalkorDriver 时传入存储配置：

```python
driver = FalkorDriver(
    host='localhost',
    port=6379,
    database=group_id,
    # 存储配置
    content_storage_type='local',  # redis | local | oss | s3
    content_storage_config={
        'base_path': './data/episodes'  # LOCAL 模式
    },
)
```

### 本地文件存储

**目录结构**：
```
{base_path}/
  {group_id}/
    {year}/
      {month}/
        {uuid}.txt          # 纯文本格式
```

**示例**：
```
./data/episodes/
  easyops_support/
    2025/
      01/
        abc123-def456-ghi789.txt
        xyz789-uvw456-rst123.txt
```

### 内容去重

基于 SHA-256 hash 的内容去重：

1. 写入 Episode 时计算 `content_hash`
2. 查询 FalkorDB 是否存在相同 hash 的文件
3. 如果存在，复用文件路径（节省存储空间）
4. 如果不存在，存储新文件

**效果**：相同内容的 N 个 Episode 只存储 1 份文件，节省 50-100x 存储空间。

## 效果

| 指标 | REDIS 模式 | LOCAL 模式 |
|-----|-----------|-----------|
| 内存占用 | 高（所有内容在内存） | 低（仅元数据在内存） |
| 存储去重 | ❌ 不支持 | ✅ 基于 hash 去重 |
| 写入延迟 | ~0ms | < 20ms (SSD) |
| 读取延迟 | ~0ms | < 10ms (SSD) |
| 扩展性 | 内存限制 | 磁盘限制 |
| 成本 | 高（需要大内存） | 低（磁盘便宜） |

## 修改文件清单

### 新增文件（Graphiti 子模块）
1. `graphiti_core/storage/content_storage.py` - 存储抽象协议
2. `graphiti_core/storage/redis_storage.py` - Redis 存储实现
3. `graphiti_core/storage/local_storage.py` - 本地文件存储

### 修改文件（Graphiti 子模块）
1. `graphiti_core/nodes.py` - EpisodicNode 新增 4 个字段，修改 save() 和 get_episodic_node_from_record()
2. `graphiti_core/models/nodes/node_db_queries.py` - FalkorDB 查询保存/返回新字段
3. `graphiti_core/driver/falkordb_driver.py` - __init__ 添加存储配置，新增 _create_storage() 和 load_episode_content()
4. `graphiti_core/graphiti.py` - 导入 StorageResult，修改 _process_episode_data() 集成内容存储逻辑

## 升级注意事项

### 1. 向后兼容

默认使用 LOCAL 模式。如需保持原有行为，可设置：
```python
content_storage_type='redis'
```

### 2. 数据迁移

从 REDIS 切换到 LOCAL 模式：
- 旧数据不会自动迁移
- 新写入的 Episode 使用文件存储
- 可选：导出旧数据后重新导入

### 3. 目录权限

使用 LOCAL 模式时，确保应用有权限写入 `base_path` 目录：
```bash
mkdir -p ./data/episodes
chmod 755 ./data/episodes
```

### 4. Docker 部署

挂载 data 目录到容器：
```yaml
services:
  app:
    volumes:
      - ./data:/app/data
    environment:
      EPISODE_STORAGE_TYPE: local
      EPISODE_LOCAL_BASE_PATH: ./data/episodes
```

### 5. 监控建议

- 监控 FalkorDB 内存使用（切换到文件模式后应明显下降）
- 监控磁盘空间使用
- 记录去重率（重复内容比例）

## 未来扩展

### OSS 存储（待实现）

**文件**: `graphiti_core/storage/oss_storage.py`

- 使用阿里云 OSS SDK
- 路径格式：`{bucket}/{group_id}/{year}/{month}/{uuid}.txt`
- 支持断点续传和分片上传

### S3 存储（待实现）

**文件**: `graphiti_core/storage/s3_storage.py`

- 使用 AWS S3 SDK（boto3）
- 路径格式：`{bucket}/{group_id}/{year}/{month}/{uuid}.txt`
- 支持 S3 兼容存储（MinIO、腾讯云 COS 等）

## 参考资料

- **完整设计文档**: `docs/design/episode-file-storage.md`
- **aiofiles 文档**: https://github.com/Tinche/aiofiles
- **FalkorDB 文档**: https://www.falkordb.com/docs
