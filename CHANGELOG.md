# 更新日志

## 目录结构更新（最新）

### 变更说明
项目目录结构已更新为符合uv和现代Python包管理工具的标准结构。

### 主要变更

1. **包结构重组**
   - 原结构：`src/model/`, `src/data/`, `src/utils/`
   - 新结构：`src/ScriptedVLA/model/`, `src/ScriptedVLA/data/`, `src/ScriptedVLA/utils/`
   - 包名：`ScriptedVLA`

2. **导入方式更新**
   - 原导入：`from src.model import ...`
   - 新导入：`from ScriptedVLA.model import ...`
   - 所有脚本文件已更新导入语句

3. **文档更新**
   - 更新了 `README.md` 中的项目结构说明
   - 更新了 `QUICKSTART.md` 中的命令示例
   - 更新了 `EXAMPLES.md` 中的路径引用
   - 添加了项目结构说明章节

4. **配置文件更新**
   - `pyproject.toml` 已配置正确的包路径：`packages = ["src/ScriptedVLA"]`

### 优势

- ✅ 符合PEP 517/518标准
- ✅ 支持uv、pip、poetry等现代包管理工具
- ✅ 便于代码组织和模块化
- ✅ 支持作为库被其他项目引用
- ✅ 安装后可直接使用 `from ScriptedVLA import ...`

### 使用方式

安装项目后（`uv pip install -e .`），可以直接使用：

```python
from ScriptedVLA.model import VLAModel
from ScriptedVLA.data import VLADataset, LIBERODataset
from ScriptedVLA.utils import load_config, setup_logger
```

### 迁移指南

如果您之前使用的是旧版本，请：

1. 更新所有导入语句：
   ```python
   # 旧方式
   from src.model import VLAModel
   
   # 新方式
   from ScriptedVLA.model import VLAModel
   ```

2. 更新命令行工具调用：
   ```bash
   # 旧方式
   python -m src.data.download_datasets --dataset libero
   
   # 新方式
   python -m ScriptedVLA.data.download_datasets --dataset libero
   ```

3. 重新安装项目：
   ```bash
   uv pip install -e .
   ```

