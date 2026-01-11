# 关于lerobot版本降级方案的说明

## ⚠️ 重要说明

**lerobot==0.3.4 版本不存在**。通过检查PyPI，发现该版本号不存在。

## 数据集格式分析

从您的数据集结构来看：
- `info.json` 位于 `meta/` 目录下（这是v3.0格式的特征）
- 数据使用 `parquet` 格式存储（`data/chunk-000/episode_*.parquet`）
- 虽然 `codebase_version` 标记为 `"v2.1"`，但实际结构是v3.0格式

**结论：您的数据集实际上是v3.0格式，应该使用最新版本的lerobot库。**

## 推荐的解决方案

### 方案1：使用最新版本的lerobot（推荐）

由于您的数据集实际上是v3.0格式（使用parquet和meta/info.json），应该使用最新版本的lerobot：

```bash
pip install lerobot
```

然后使用我们之前修改的代码，代码已经包含了版本自动检测功能，会自动识别v3.0格式并使用相应的加载器。

### 方案2：如果确实需要支持真正的v2.1格式

真正的v2.1格式数据集应该：
- `info.json` 在根目录（不在meta/目录下）
- 使用 `episode_*.hdf5` 文件（不是parquet格式）

如果您有这样的数据集，可以使用我们创建的自定义加载器（LeRobotDatasetV21类），代码中已经包含了这个功能。

## 当前数据集的处理

对于您当前的数据集（虽然是v2.1标记，但实际是v3.0格式），建议：

1. **使用最新版本的lerobot**：
   ```bash
   pip install lerobot
   ```

2. **使用现有的代码**，代码已经支持版本自动检测：
   - 代码会自动检测数据集版本
   - 对于v3.0格式（meta/info.json + parquet），会使用LeRobotDataset类
   - 对于真正的v2.1格式（根目录info.json + hdf5），会使用LeRobotDatasetV21类

3. **在config.yaml中设置**：
   ```yaml
   lerobot_test:
     dataset:
       local_path: "./dataset/libero_object/"
       dataset_version: "auto"  # 自动检测
   ```

## 总结

- ❌ 降级到0.3.4不可行（版本不存在）
- ✅ 您的数据集是v3.0格式，应该使用最新版本的lerobot
- ✅ 代码已经支持版本自动检测，可以直接使用
