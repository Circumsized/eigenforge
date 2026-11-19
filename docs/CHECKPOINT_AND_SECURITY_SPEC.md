# 崩溃一致性持久化、产物溯源与安全规范（Checkpoint & Security Specification）

本文档阐述 `eigenforge` 在模型权重反序列化安全、耐久原子文件写入、崩溃恢复 Checkpoint 协议及产物全链路溯源上的工程设计与代码契约。

---

## 1. PyTorch 反序列化安全版本守卫（`src/eigenforge/common/torch_guard.py`）

### 1.1 威胁模型与双 CVE 版本底线

PyTorch 的默认 `.pt` / `.pth` 权重格式基于 Python `pickle`。历史上即便显式传入 `torch.load(..., weights_only=True)`，在旧版本 PyTorch 上仍存在两处高危安全绕过漏洞（细节见表 1）：

**表 1　PyTorch weights_only 反序列化已知漏洞与修复版本**

| 漏洞编号 / GHSA | 严重程度 | 受影响版本范围 | 首个修复版本 | 漏洞机理说明 |
| :--- | :---: | :---: | :---: | :--- |
| **CVE-2025-32434**<br>`GHSA-53q9-r3pm-6pq6` | Critical (CVSS v4 9.3) | `torch <= 2.5.1` | `2.6.0` | `weights_only=True` 受限反序列化器在旧版本上仍可被恶意构造的权重文件触发远程代码执行（RCE） |
| **CVE-2026-24747**<br>`GHSA-63cw-57p8-fm3p` | High | `torch < 2.10.0`<br>(`<= 2.9.1`) | **`2.10.0`** | `weights_only` 专用 unpickler 在解析畸形指令流时发生内存破坏，可导致任意代码执行 |

因此，本仓库将运行时安全底线强制设定为 `MIN_SAFE_TORCH = (2, 10, 0)`，对任何版本字符串 $v$ 解析为可比较四元组：

$$
\operatorname{parse\_version}(v) = \big(\text{major}, \; \text{minor}, \; \text{patch}, \; p\big), \qquad p = \begin{cases} -1, & v \text{ 为 } \text{a/b/rc/dev 预发布版} \\ 0, & v \text{ 为正式稳定版} \end{cases}
\tag{1.1}
$$

仅当 $\operatorname{parse\_version}(v) \ge (2, 10, 0, 0)$ 且 $p = 0$ 时判定为安全运行时。

### 1.2 双层防线（静态 AST 扫描 + 运行时版本门控）

两道防线的分工与判定路径见图 1；版本解析规则 $\operatorname{parse\_version}(v)$ 见式 (1.1)。

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 80, 'rankSpacing': 85, 'padding': 16, 'diagramPadding': 14}}}%%
flowchart TB
    subgraph static["　　① 第一道防线: 静态 AST 语法树扫描 (tests/test_torch_guard.py)　　"]
        direction TB
        AST1(" 　<br/>　 遍历全仓源码　 <br/> _product_sources 动态发现所有产品代码 ")
        AST2(" 　<br/>　 追踪所有 torch.load 导入别名　 <br/> import torch · import torch as t · from torch import load ")
        AST3(" 　<br/>　 静态安全断言　 <br/> ① 唯一允许调用点 = checkpoints.py <br/> ② 必须携带字面量 weights_only=True ")
        AST1 --> AST2 --> AST3
    end

    subgraph runtime["　　② 第二道防线: 运行期版本门控 (common/torch_guard.py)　　"]
        direction TB
        RT1(" 　<br/>　 启动与加载门禁调用　 <br/> enforce_safe_torch_at_startup / require_safe_torch ")
        RT2{"torch 版本 >= 2.10.0<br/>且为正式稳定版?"}
        RT_OK(" 　<br/>　 安全准入放行　 <br/> 免疫 CVE-2025-32434 与 CVE-2026-24747 ")
        RT_ENV{"环境变量 EIGENFORGE_ALLOW_UNSAFE_TORCH == 1?<br/>(仅读进程环境，忽略 .env)"}
        RT_WARN(" 　<br/>　 双通道告警放行　 <br/> stderr + logger.warning 记录风险后继续运行 ")
        RT_FAIL(" 　<br/>　 拒绝执行 (退出码 2)　 <br/> 抛出 UnsafeTorchError 阻断加载 ")
        RT1 --> RT2
        RT2 -- "是 (正式版 >= 2.10.0)" --> RT_OK
        RT2 -- "否 (< 2.10.0 或预发布)" --> RT_ENV
        RT_ENV -- "是 (显式声明)" --> RT_WARN
        RT_ENV -- "否 (默认安全)" --> RT_FAIL
    end

    static ~~~ runtime

    style static fill:none,stroke:#1a73e8,stroke-width:2px,stroke-dasharray: 4 4
    style runtime fill:none,stroke:#c25e00,stroke-width:2px,stroke-dasharray: 4 4
    style RT_OK stroke:#238636,stroke-width:2.5px
    style RT_FAIL stroke:#d73a49,stroke-width:2.5px
    style RT_WARN stroke:#c25e00,stroke-width:1.5px
```

**图 1　torch.load 双层安全防线（静态 AST 扫描 + 运行期版本门控）**

1. **静态 AST 全仓扫描守卫（`tests/test_torch_guard.py`）**：
   - 动态遍历仓库所有产品源码，通过 AST 语法树追踪 `import torch`、`import torch as t` 与 `from torch import load` 等所有导入别名；
   - 强制断言：**全仓唯一允许调用 `torch.load` 的文件是 `src/eigenforge/checkpoints.py`**，且每一次调用必须携带字面量关键字参数 `weights_only=True`。
2. **运行时版本门控（`require_safe_torch` & `enforce_safe_torch_at_startup`）**：
   - 所有训练与遍历 CLI 在 `parse_args()` 之后、触碰任何磁盘文件之前调用 `enforce_safe_torch_at_startup()`；`load_checkpoint()` 与 `GenAIEvaluator._load_inception()` 内部亦无条件调用 `require_safe_torch()`；
   - 当运行时 PyTorch 低于 `2.10.0` 时，CLI 捕获 `UnsafeTorchError` 并以**退出码 `2`** 失败封闭退出。
3. **受审计的逃生舱口（`EIGENFORGE_ALLOW_UNSAFE_TORCH=1`）**：
   - 仅从**进程环境变量 `os.environ`** 读取（绝不从 `.env` 文件读取，防止克隆含恶意 `.env` 的目录自动放行）；
   - 仅当值严格等于字符串 `"1"` 时放行，且放行时强制通过**双通道**（`sys.stderr` 标准错误流 + `logger.warning` 持久化日志文件）输出包含两个 CVE 编号的显式告警。

---

## 2. 耐久原子文件替换原语（`src/eigenforge/common/fsutil.py`）

直接向目标路径覆写文件（如 `torch.save(obj, "latest.pt")`）或仅调用 `os.replace` 存在两类数据损坏风险：

1. 写盘中途进程被杀（SIGINT / OOM）会直接截断唯一的一份 `latest.pt`；
2. 在 ext4 等延迟分配（delayed allocation）文件系统上，若不先对文件数据执行 `fsync`，断电后重命名目录项可能先于数据块落盘，产生**长度为 0 的空文件**。

`durable_replace` 的六步写序与异常回滚路径见图 2。

### `durable_replace` / `durable_replace_bytes` 六步写序流程图

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 80, 'rankSpacing': 85, 'padding': 16, 'diagramPadding': 14}}}%%
flowchart TB
    subgraph forward["　　① durable_replace 正常六步持久化写序 (Happy Path)　　"]
        direction LR
        Step1(" 　<br/>　 ① 检查目录并打开 dir_fd　 <br/> mkdir(parents=True) → POSIX 打开父目录 <br/> 「Windows 无目录 fsync，返回 None」 ") --> Step2(" 　<br/>　 ② 同目录创建隐藏临时文件　 <br/> .{target}-{rand}.tmp ")
        Step2 --> Step3(" 　<br/>　 ③ 完整写入字节数据　 <br/> 「UTF-8 + newline='' / 二进制 wb」 ")
        Step3 --> Step4(" 　<br/>　 ④ 强制刷盘并关闭句柄　 <br/> handle.flush() → os.fsync(fd) ")
        Step4 --> Step5(" 　<br/>　 ⑤ 原子重命名就位　 <br/> os.replace(tmp_name, target) ")
        Step5 --> Step6(" 　<br/>　 ⑥ 刷新父目录项元数据　 <br/> POSIX: os.fsync(dir_fd) → close ")
    end

    subgraph failure["　　② 异常回滚与残留清理机制 (Failure Rollback)　　"]
        Rollback(" 　<br/>　 异常安全回滚分支　 <br/> unlink(tmp_name) 移除未完成临时文件 <br/> 原 target 目标文件保持逐字节完好无损 ")
    end

    forward ~~~ failure

    Step2 -. "mkstemp / fdopen 失败" .-> Rollback
    Step3 -. "写入回调抛出异常" .-> Rollback
    Step4 -. "fsync EIO 物理故障" .-> Rollback
    Step5 -. "replace 重命名失败" .-> Rollback

    style forward fill:none,stroke:#238636,stroke-width:2px,stroke-dasharray: 4 4
    style failure fill:none,stroke:#d73a49,stroke-width:2px,stroke-dasharray: 4 4
    style Step1 stroke:#1a73e8,stroke-width:1.5px
    style Step4 stroke:#238636,stroke-width:2px
    style Step5 stroke:#238636,stroke-width:2px
    style Step6 stroke:#8a63d2,stroke-width:1.5px
    style Rollback stroke:#d73a49,stroke-width:2px
```

**图 2　durable_replace 六步持久化写序与异常回滚**

---

## 3. 崩溃一致性 Checkpoint 提交与恢复协议（`src/eigenforge/checkpoints.py`）

### 3.1 两阶段保存与 `.prev` 硬链接轮转时序图（`save_checkpoint`）

由于操作系统无法跨两个文件（`latest.pt` 与 `latest.pt.manifest.json`）提供原生双文件事务，`save_checkpoint()` 通过两阶段提交协议保证崩溃可恢复性（完整交互时序见图 3）：

```mermaid
%%{init: {'sequence': {'actorMargin': 100, 'messageMargin': 55, 'noteMargin': 16, 'boxMargin': 14, 'mirrorActors': false}}}%%
sequenceDiagram
    autonumber
    participant Trainer as 训练主循环 (train_*.py)
    participant Circuit as 熔断器 (check_finite_*)
    participant Ckpt as save_checkpoint
    participant Mem as 内存缓冲 (io.BytesIO)
    participant Prev as 上一代备份 (.prev 硬链接)
    participant Disk as 磁盘原子写 (fsutil)

    Trainer->>Circuit: check_finite_parameters(modules, epoch)
    Circuit-->>Trainer: 全部参数与 Buffer 有限 (否则抛 NonFiniteLossError 拒存)
    Trainer->>Ckpt: save_checkpoint(latest.pt, payload, config, seed, step)

    Note over Ckpt,Mem: 阶段一：纯内存预检（零磁盘副作用）
    Ckpt->>Mem: torch.save(payload, buffer) 序列化为字节流 data
    Ckpt->>Mem: 计算 sha256(data) 并构造 ArtifactManifest
    Ckpt->>Mem: manifest.to_json_text() 校验 JSON 可序列化性

    Note over Ckpt,Disk: 阶段二：保留上一代 → 写数据 → 写清单
    Ckpt->>Prev: os.link(latest.pt → .latest.pt.prev.link) → os.replace(latest.pt.prev)
    Ckpt->>Prev: os.link(manifest → .latest.pt.prev.manifest.json.link) → os.replace(latest.pt.prev.manifest.json)
    Ckpt->>Disk: durable_replace_bytes(latest.pt, data) [flush + fsync + replace]
    Ckpt->>Disk: durable_replace(latest.pt.manifest.json, json_text)

    Ckpt-->>Trainer: 返回已落盘的 ArtifactManifest
```

**图 3　save_checkpoint 两阶段提交与 .prev 硬链接轮转时序**

### 3.2 同旁溯源清单 `ArtifactManifest` 与 Git 元数据（`src/eigenforge/artifacts.py`）

每次保存生成的 `<artifact>.manifest.json`（`manifest_schema_version = 1`）包含以下字段：

```json
{
  "manifest_schema_version": 1,
  "artifact_path": "experiments/eigengan/latest.pt",
  "artifact_sha256": "<64-char-lowercase-hex-sha256>",
  "model_kind": "eigengan",
  "seed": 42,
  "step": 6350,
  "training_config": { "z_dim": 128, "image_size": 64, "n_basis": 6, "..." : "..." },
  "dataset": { "name": "celeba", "root": "./data/celeba", "num_images": 202599 },
  "metrics": null,
  "git": { "commit": "<40-char-commit-sha>", "dirty": false },
  "created_utc": "2026-11-18T06:00:00Z"
}
```

### 3.3 单次读入防 TOCTOU 校验与故障回退决策图（`load_checkpoint`）

完整的验证流水线、`require_manifest` 两种取值下的行为差异与 `.prev` 回退条件见图 4 的决策树：

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 70, 'rankSpacing': 85, 'padding': 16, 'diagramPadding': 14}}}%%
flowchart TD
    Start(" 　<br/>　 调用 load_checkpoint(path)　 <br/> 恢复加载主入口 ") --> Guard["① require_safe_torch()<br/>强制校验 torch ≥ 2.10.0"]
    Guard -- "版本不安全且无逃生变量" --> ErrTorch(" 　<br/>　 拒绝执行 (UnsafeTorchError)　 <br/> 防御 CVE-2025-32434 与 CVE-2026-24747 ")
    Guard -- "安全通过" --> CheckFile{"② 检查主文件与同旁清单"}

    CheckFile -- "主文件缺失 或 清单损坏/非普通文件" --> ErrInteg(" 　<br/>　 完整性异常 (CheckpointIntegrityError)　 <br/> 触发 .prev 回退机制 ")
    CheckFile -- "无同旁清单 且 require_manifest=True" --> ErrInteg
    CheckFile -- "无同旁清单 且 require_manifest=False" --> WarnNoMan(" 　<br/>　 遗留兼容模式 (无清单)　 <br/> logger.warning 告警后继续读入 ")
    CheckFile -- "清单解析成功" --> SingleRead(" 　<br/>　 ③ 单次读入内存: data = target.read_bytes()　 <br/> 消除校验与反序列化之间的 TOCTOU 文件调包窗口 ")

    SingleRead --> CheckHash{"④ sha256(data) == manifest.sha256?"}
    CheckHash -- "哈希不匹配 (写入中断或被篡改)" --> ErrInteg
    CheckHash -- "哈希严格匹配" --> CheckDrift{"⑤ expected_config 与 manifest 比对"}

    CheckDrift -- "超参不一致" --> ErrSchema(" 　<br/>　 架构漂移 (CheckpointSchemaError)　 <br/> 配置漂移属人为传参错误，绝不回退 .prev ")
    CheckDrift -- "架构一致" --> Unpickle(" 　<br/>　 ⑥ 内存反序列化 (同一份字节)　 <br/> torch.load(io.BytesIO(data), weights_only=True) ")
    WarnNoMan --> Unpickle
    Unpickle -- "缺少必需键或非法 epoch" --> ErrSchema
    Unpickle -- "验证全部通过" --> Success(" 　<br/>　 加载成功 (ResumeState)　 <br/> 返回 payload, epoch, manifest ")

    ErrInteg --> FallbackCheck{"fallback_to_previous=True<br/>且存在 latest.pt.prev?"}
    FallbackCheck -- "是" --> RetryPrev(" 　<br/>　 启动 .prev 上一代回退　 <br/> logger.warning 告警并切换目标为 latest.pt.prev ")
    RetryPrev -- "latest.pt.prev 验证通过" --> Success
    RetryPrev -- "latest.pt.prev 亦损坏" --> Abort(" 　<br/>　 终止恢复 (抛出异常)　 <br/> 报告不可恢复错误 ")
    FallbackCheck -- "否" --> Abort

    style Start stroke:#1a73e8,stroke-width:2px
    style SingleRead stroke:#238636,stroke-width:2px
    style Unpickle stroke:#238636,stroke-width:2px
    style Success stroke:#0891b2,stroke-width:2.5px
    style RetryPrev stroke:#c25e00,stroke-width:2px
    style WarnNoMan stroke:#c25e00,stroke-width:1.5px
    style ErrTorch stroke:#d73a49,stroke-width:2px
    style ErrSchema stroke:#d73a49,stroke-width:2px
    style Abort stroke:#d73a49,stroke-width:2px
```

**图 4　load_checkpoint 防 TOCTOU 校验与 .prev 回退决策**

### 3.4 双层数值健康熔断器（`check_finite_losses` & `check_finite_parameters`）

1. **步级损失熔断 `check_finite_losses(losses, epoch=..., step=...)`**：
    - 时机：在每次 `loss.backward()` / `optimizer.step()` **之前**。
    - 行为：检查损失张量是否全部有限（`torch.isfinite`）；一旦出现 `NaN` 或 `Inf`，立即抛出 `NonFiniteLossError` 终止训练，防止坏梯度污染模型权重。
2. **轮级参数熔断 `check_finite_parameters(modules, epoch=...)`**：
    - 时机：在每个 Epoch 结束、调用 `save_checkpoint()` 之前。
    - 范围：遍历生成器、EMA 生成器与判别器的全部浮点参数（`parameters()`）与缓冲区（`buffers()`）。
    - 行为：若最后一步优化器更新或 AMP 缩放导致任何权重出现 `NaN/Inf`，立即抛出 `NonFiniteLossError` 拒绝写盘——磁盘上的 `latest.pt` 永远不会被爆炸的模型覆盖。

---

## 4. 纯张量权重安全互操作（`src/eigenforge/safetensors_io.py`）

针对仅需分发或加载生成器/EMA 纯张量权重的场景，`safetensors_io.py` 提供零 Pickle 执行路径的互操作封装（需安装可选依赖 `pip install -e ".[safetensors]"`）：
- `export_state_dict(state_dict, path, *, metadata=None)`：校验所有值均为 `torch.Tensor`（非张量对象立即抛 `TypeError`），转为 CPU 连续内存后在内存序列化，并经由 `durable_replace_bytes` 原子持久化；
- `import_state_dict(path, *, device="cpu")`：通过 `safetensors.torch.load_file` 安全加载张量字典；
- `metadata(path)`：通过 `safe_open` 只读解析文件头部元数据字典，无需加载完整张量权重。
