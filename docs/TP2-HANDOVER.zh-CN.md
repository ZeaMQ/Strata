# Strata 张量并行（TP=2）——半成品交接说明

**基线版本：Strata 0.1.44.1，commit `296b59d2`（2026-10-14），CUDA 源码构建，Windows。**
本文所有数字都在该基线上测得；换一个 commit 可能无法干净套用。

**来源说明：** 本工作由 AI 辅助，在**一次超过 400 轮对话、4 000 步工具调用**的会话中完成
（会话自身计数），**未经人工 review**。第 4 节描述的快路径经过 **13 次** 独立修复尝试仍未解决；
第 14 次把三个 KV 池的分配方、写入方、读取方全部列成一张表对账，找到了根因。请把它当作"附带补丁的 bug 报告"，而不是一个提交。

**状态：半成品，但可用。** 双卡张量并行路径存在，在我测过的每一个 prompt 长度上都**正确**。
本文首次发布时它的快速路径是坏的，**2026-10-19 已修复**（commit `7f55fd3`，见第 0 节）：prefill 现在
1 187–1 519 tok/s，约为层切分的一半——实验可用，但仍超不过层切分。第 4、4a、5 节保留为排查记录。
**请不要以当前形态发布。**

若仓库已有张量并行的讨论线程，请把本文并入那条线程，而不是另开新 issue。

---

## 0. 已解决（2026-10-19）：快路径 non-finite logits 的根因与修复

commit `7f55fd3`（"fix(TP): the batched prompt path's K/V append wrote the model's page stride into the
rank's pools"）。下面第 4、4a、5 节是排查过程的记录；排查到此为止。

**根因，一段话说清。** batched 快路径里唯一不一致的组件是它的 K/V 写入方。常驻 slot 池、pinned host
副本、staging 池全部按 `qsa_shapes_rank` 分配（TP=2 时每页 **1** 个 KV head），所有读取方——attention
内核（`(page * n_kv_heads + kvh) * page_size`）、`kv_stream` 的块搬运、gather、decode 内核——都按这个
stride 寻址。只有 `kv_append_kernel` 还在按模型的每页 2 个 head、外加 rank 的**全局** head 偏移写入：
每一行都落在读取方所用偏移的两倍处，且该 rank 靠后的页直接越过池尾。attention 于是读到从未写过的行；
从未写过的 INT8 行 scale 为 0，反量化要除以它——0/0——第一个 QSA 层的 attention 就此 non-finite，
#879 在 prompt 末尾引爆。verify 窗口路径从不调用这个内核，所以它一直是对的。

**为什么 13 次都错过了。** 每次都只改一对里的一侧而留下另一侧。把写入改成 `page*1` 时 `head_base` 仍是
全局的——rank 1 的行被搬进 rank 0 的下一页，这就是第 5 节里"100% non-finite、max|x|=0"那次反转，
它错误地洗清了写入方的嫌疑。答案是那张对账表掉出来的，不是再探针探出来的。

**修复后实测**（同机、同配置、`STRATA_TP_FASTPROMPT=1`、专家缓存已热）：2 803 / 5 533 / 11 176 token 的
prompt 全部完整读入，引擎日志中没有任何 non-finite；稳态 1 187 与 1 519 tok/s（首个大 prompt 付约 86 s
图捕获）；decode 约 90 tok/s；`12*7` 答 `84`，摘要与推理回答的思考与生成端到端连贯正确。快路径现在是
我会用的路径；verify 窗口保持默认，作为安全兜底。

**修复后仍然成立：** MoE 在两个 rank 上各算一遍，TP prefill（~1.5 K）仍低于层切分（2.1–2.9 K）——
4a 的第二阶段分析不变。仍未测：Q4_0 / K8V4 的 batched append（它们还在用模型级形状）、HIP、SYCL、
长时间多轮会话。

---

## 1. 这是什么

Strata 目前的多卡方式是**切层**（`--pipeline-windows 2`、`layer_split: auto`）：每张卡拥有整层，
不复制任何专家。本工作加入第二种模式 `--tp 2`：**只切分 dense/attention 权重，MoE 在两卡上各留一份**。
这个实验的目的是测出这种切法在这台机器上能换来什么。目前的答案是：**正确，但没更快**。

## 1a. 为什么还要做：因为 #156

上游**已经否决过**这个功能：issue **#156「[Feature Request] support tensor parallelism for multi gpu
setup」**于 2026-09-30 被关闭，理由是*「暂不计划：decode 受专家限制，切 dense 部分的张量并行在此收益很小」*。
基线 `docs/MULTI_GPU.md` 也写了一句话：*「这是流水线（层）并行，**不是张量并行**」*；而基线 `296b59d2`
里没有 `--tp`、没有 `TpRuntime`、也没有 `shard.hpp` —— 整套路径是新代码，不是对已有实现的修改。

**所以本文不是功能提案，而是 #156 当初缺的那份实测。** 我把东西做出来是为了测量，而下面的数字
**支持维护者关闭它的理由**：

- 端到端，正确的 TP 路径约为层切分 prefill 吞吐的 **2%**（同一机器、同一 pack、同一天：37–52 tok/s
  对 2 114.8 tok/s）；
- 分段计时说明了原因：**host grouping 占 TP prefill 的 47.8%**——两个 rank 把专家路由到同一个 CPU
  pool——再加上 `gdn out proj`（TP 的 reduce）再占 16.2%。

因此真正有价值的产出是**一个带证据的否定结果**，以及产生它的代码。如果维护者仍然不想要，有用的
部分是第 8 节那两个真 bug。

## 2. 硬件与软件（下面每个数字的测量环境）

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 9 5900X，12 核 / 24 线程，`--pool-workers 11` |
| 内存 | 80 GiB |
| GPU 1、2 | 2× NVIDIA GeForce RTX 3080，各 20 480 MiB，PCIe 4.0 x8 |
| GPU 0 | Quadro P1000 4 GB——仅视觉编码器（`vision.cuda_device: 2`） |
| 操作系统 | Windows 11（10.0.26200） |
| Strata | 引擎 **0.1.44.1**，基线 commit **296b59d2**，CUDA 源码构建 |
| 模型 | Qwen3.8-Flash-Next abliterated **IQ3_S**，pack `sc117-iq3_s`，native experts |
| 关键参数 | `--max-context 524288 --kv int8 --kv-resident 32768 --rope-scaling yarn --rope-scale 2 --expert-cache auto --expert-profile <profile> --vram-reserve-mib 64 --vram-reserve-later-mib 8 --spec 4 --prefill auto:16384 --vision` |
| TP 必需环境变量 | `STRATA_SPLIT_MTP_BATCH=1`、`STRATA_GDN_KEYHEAD=0`（后者**必需**，见第 6 节） |

## 3. 能用的部分（实测）

`--tp 2`，prompt 走两 rank 的 verify 窗口（本次补丁给 TP 设的默认路径）：

| prompt token | 读取 | prefill | decode |
|---|---|---|---|
| 108 | 108 / 108 | 43.6 tok/s | 37.4 tok/s |
| 3 458 | 3 458 / 3 458 | 37.7 tok/s | 45.9 tok/s |
| 6 856 | 6 856 / 6 856 | 52.1 tok/s | 58.4 tok/s |
| 14 060 | 14 060 / 14 060 | 41.3 tok/s | 43.4 tok/s |

- 上述长度均无非有限 logits；`12*7` 返回 `84`；能完整生成。
- 每卡显存 **19 909 / 19 867 MiB（19.44 / 19.40 GB）**，与层切分在这个 pack 上的 19 399 / 19 957 MiB 持平，
  这是本次的目标之一。
- 视觉、MTP 草稿层、`--kv-resident`、`--spec` 在该模式下均可用。

**对照：同一台机器、同一 pack 上现有的层切分**（同一天早先测得）：36 329 token 的 prompt
**2 114.8 tok/s**，decode **99.3 tok/s**，显存 19 399 / 19 957 MiB。
也就是说，正确的 TP 路径大概是层切分 prefill 吞吐的 **2%**。这是诚实的结论：该补丁没有推进速度。

## 4. 不能用的部分

> **2026-10-19 已解决——见第 0 节。** 以下保留为排查记录。

快路径（`STRATA_TP_FASTPROMPT=1`，用一次批量 prefill 代替 6 token 一格的 verify 窗口）在同样的 prompt 上
能到 **860–990 tok/s**，但会产生非有限 logits：

```
strata serve: non-finite logits (#879) in a prompt window at position 6851 (row 0)
strata serve: prompt 6856 tokens = 0 reused + 6851 of 6856 read in 7972 ms (859.4 tok/s)
```

它对**能落在一个 KV page 内的 prompt** 是正确的，超过就坏。因此本补丁把 **verify 窗口留作 TP 默认**，
把批量路径改为 opt-in：

```cpp
if (o.tp > 0 && std::getenv("STRATA_TP_FASTPROMPT") == nullptr) return true;   // 走窗口
```

## 4a. 两阶段方案：先「可用的 990」，再「追平 PP 的 2900+」
>
**前提：达到 1000 不需要新的优化——它已经达到过（860–990，最高一次 1012）。需要的是让它不崩。**
所以下面的方案全部是"修正确性"，没有一条是"提速"。
>
### 快路径与默认路径差在哪
默认路径把 prompt 拆成 6 个 token 一格的 verify 窗口；快路径用**一次批量 prefill** 处理整个 prompt。
二者唯一的实质差别是 **KV 怎么写进去**：
- 批量路径：`kv_append`（预填充内核），一次写 N 个 cell
- 窗口路径：`kv_append_q8_step`（单 token 内核），每格写 5–6 个 cell
而 decode 与窗口的**读取只有一种算法**。两条写入逻辑的页行不一致，读取就必然踩空。
>
### 为什么现在不能用
批量写完 N−5 个 token 后，第一次读取（short-read 窗口或首个 decode）拿到的不是那 N 个 cell：
```
layer 3  qsa attn  583872 of 635904 non-finite (max |x| 2.288e+06)
```
而 `qsa q(in)` 探针干净——**查询是好的，坏在读 KV**。
>
### 修复顺序（按成本从低到高）
1. **对表**（预计一次改动即可定案）：在 `kv_append_kernel` 里打印前 8 个 cell 的 `row_id`，
   在 `src/kernels/cuda/qsa_prompt_attn.cu` 的 `srow` 填充处打印同 8 个 cell 的 `r`。
   两者不一致 → 根因确定。
2. **查 `ident_table`**：`prefill.cpp:1204` 构建它，`staged=1` 时读端（`pools_of(m.stage, m.ident_table)`）
   用的就是它。确认 `ident[i] == i`，而不是别的排列。
3. **记住 `page_size == 4`**：block 数 = cells/4；任何按 64 推的结论都要重算（我早前一族推断因此全错）。
4. 若第 1 步对表**一致**但输出仍坏，则问题在 stage pool 的**填充**（`kv_stage_from_host` 与
   `kv_append` 的 `stage` 参数写入），而不在行号计算——这条分支我改动过但没有实测收敛。
>
### 已知会失败的两条路（别重复）
- **只改写端页步长**（`page*2` → `page*1`）：实测从 92% 非有限变成 **100%** 非有限（`max|x|=0`）。
  反向变化说明读端寻址不是我以为的那样，写端不是杠杆。
- **只改读端 `QsaShapes`**：行为不变（计数与偏移本来就对）。
>
### 第二阶段：从 ~990 追到 PP 的 2900+（990 不是终点，只是 34%）

**参照系说明**：作者实测原版层切分（PP）的 prefill 可上 **2900+ tok/s**；本报告在同一台机器、同一 pack
上按 36 329 token 测到的是 **2 114.8 tok/s**（prompt 长度不同，读数不同，两个数都记在这里）。
单卡跑同一条 3 458-token prompt 是 **397 tok/s**。

所以：
- **~990 tok/s 只等于一张卡的产出**，第二张卡在 prefill 上几乎没被用上；
- 相对 2900 的目标，990 只有 **34%**；
- 而相对本报告测到的 2 114.8，990 是 **47%**。

**为什么只有一张卡的量**：MoE 是复制的。两个 rank 算**同一批专家**，还挤**同一个 11 线程 CPU pool**。
分段计时（6 856 token，GPU 合计 7 161 ms）：

| 阶段 | 占比 | 是否重复劳动 |
|---|---|---|
| host grouping | **47.8%** | ✅ 两 rank 完全相同的工作 |
| gdn out proj（TP reduce） | **16.2%** | ❌ TP 独有开销（层切分没有） |
| qsa proj | 7.4% | 已切分 |
| gemm gate/up + down | 6.9% | ✅ 重复 |
| gdn（conv + recurrence） | 5.4% | 已切分 |
| combine | 4.4% | ✅ 重复 |
| qsa attn | 3.3% | 已切分 |
| hc read | 4.7% | 已切分 |
| 其余（router/gather/dequant/ple/kv） | 约 4% | 混合 |

**约 60% 的 prefill 是重复劳动**，另外还背了 16.2% 的 reduce。

**关键算术（诚实结论）**：即使按专家 id 完美切分、把重复的 60% 砍半，时间也只降到约 0.70 倍，
即 **990 → 约 1 400 tok/s（约 1.4 倍）**。要到 **2 900 需要 2.9 倍**，这要求重复劳动几乎归零、
且 16.2% 的 reduce 也被吸收——**在"复制 MoE"这个设计下做不到**。

**所以阶段二不是本补丁的收尾，而是另一次设计选择：**

1. **rank 1 复用 rank 0 的分组结果**（低风险，收益最大项）。两 rank 路由到**完全相同**的专家集合，
   重复跑 host grouping 是纯浪费。理论上能砍掉接近一半的 47.8%。**未实测。**
2. **合并同层两次 reduce**（低风险）。`ssm_out` 与 `attn_output` 各一次，合计 16.2%，可合批或与下一层重叠。
3. **按专家 id 切分 + all-to-all 路由**（高风险，唯一能把重复的 60% 真正减半的做法）。
   做完大致落在 **1 400–1 800**。
4. **要达到 2 900 同级**，得让"每张卡每 token 的专家工作量"也减半——即真正把专家集合切成两半、
   每 token 只在自己拥有的专家上计算。这是与层切分并列的第三种并行方式，**不是 TP 的补丁**。

**一句话**：修 KV 只能把你从 0 带到 990；1、2、3 三条约到 1 400–1 800；
**2 900 需要第 4 条，那是重新设计而不是修 bug。**

### 修好之后能到多少
批量路径实测 **860–990 tok/s**，即层切分（2114.8 tok/s）的 **41–47%**。这是复制式 MoE 的上限附近：
两 rank 把同一批专家路由到**同一个 11 线程 CPU pool**，分段计时里 **host grouping 占 47.8%**。
所以要继续往上，必须动专家调度（或改成按专家 id 切分 + all-to-all 路由），**不是**继续修 KV。

## 5. 证据链（全部为实测）

1. **单卡对照：同一二进制、同一 prompt。** 单卡 3 458 token：3 458/3 458 读完、生成 40 token、无错误。
   双 rank：3 453/3 458，随后尾部窗口出错。单卡**根本不存在**"批量写入 → 窗口读取"这条衔接，TP 有。
2. **TP 强制走窗口**（`STRATA_TP_WINDOWS=1`）：3 458/3 458 正常——除批量 prompt 路径外的一切
   （reduce、GDN、MoE、显存）都是好的。
3. **`STRATA_DBG_NAN=1`**（原有）：报 layer 1/2 的 MoE 整块非有限（`GU` 3453×12800 全 NaN）。
   MoE 在下游，这是症状。
4. **`STRATA_DBG_GDN=1`**（本补丁新增）：给出真实断面

   ```
   layer 0/1/2  GDN 块        干净
   layer 3      qsa q(in)     干净            ← 查询本身是好的
   layer 3      qsa attn      583872/635904 非有限（max |x| 2.288e+06）
   layer 3      qsa bo(proj)  527360/529920 非有限
   layer 3      qsa bo(reduce) 527360/529920 非有限
   layer 4      GDN 块        整块非有限（下游）
   ```

   GDN 的计数与偏移本身是对的（`ghv=24 gch=5120 ghk=8 tpr=2`，`state_off` 严格为 408 576 的整数倍），
   所以 GDN 的状态切片没有重叠。
5. **从运行中的内核路径打出的配置**：
   ```
   layer 3  staged=1  kv_mode=1  page_size=4  s.n_head_kv=2
   ```
   其中两点代价最大：
   - **`page_size` 是 4，不是 64。** 我早前一大族推断都假设 64，全部作废。
   - **`staged=1`**：prompt 注意力读的是 **stage pool + `m.ident_table`**，不是 session pool。

**我有信心的结论**：查询是好的，故障在 QSA 层如何读取 KV cell，且与"批量 append + staged pool"这一
组合绑定。

### 已试过并回退的（避免重复劳动）

页步长两个方向都试过，单独或与读端一起改都无效：写入 `page*2` / 读取 `page*1` 得到 92% 非有限
（`max |x| 2.29e6`），写入 `page*1` / 读取 `page*1` 得到 **100%** 非有限（`max |x| 0`）。
这个反向变化是最强的线索：**说明读端的寻址不是我以为的那样，写端不是杠杆。**

## 6. 一定不能忘的一件事

TP 下 **`STRATA_GDN_KEYHEAD=0` 是必需的**。`gdn_rec_kh_kernel` 未被本补丁参数化，仍按模型的
全量 `HV=48 / HK=16` 运行，而一个 rank 只有 24 / 8。不设该环境变量，GDN 会写越自己的状态切片。
在把这个模式变成默认之前，那个内核需要和第 7 节的几个一样处理。

## 7. 本补丁改了哪些文件，为什么

| 文件 | 改动 |
|---|---|
| `include/strata/prefill/kernels.hpp` | `gdn_gates` / `gdn_conv` / `gdn_recurrence` / `split_q` / `gate_attn` / `kv_append` 增加运行时的 head/通道数参数，默认仍为模型全量 |
| `src/prefill/kernels.cu` | 上述内核使用这些参数。**`kv_append` 仍故意使用模型的 2-head 页行（`page*2`）**——rank 本地行（`page * heads + kvh`）是被试过并让第一个 QSA 层注意力变成 100% 非有限的方案，已回退。`CONV_TILE` 未改 |
| `src/prefill/prefill.cpp` | GDN 块传入 rank 的计数；QSA 注意力改用 rank 局部 `QsaShapes` 与 `qh*256` 的 query 步长；`ssm_out` 与 `attn_output` 各加一次 TP reduce；KV staging 腿接收 rank 局部 shapes；新增 `STRATA_DBG_GDN` 探针 |
| `src/program/generate.cpp` | TP 的 prompt 路径选择（默认窗口、批量 opt-in）；专家缓存重试改为按 `device_free_bytes()` 填充而不是固定 3/4——**这是显存从 18.4 到 19.44 GB 的唯一原因** |
| `src/core/verify.cpp` / `src/core/layer.cpp` / `src/core/session.cpp` | verify 侧的 rank 局部 GDN/QSA shapes |

**给 review 者的警告：** 基线 `296b59d2` 里完全没有张量并行的代码，所以这个分支是**一份未提交工作树的
快照**，不是一串经过 review 的 commit。它包含整个 TP 工作——19 个修改文件加 20 个新文件
（`include/strata/core/{shard,native_shard,tp_runtime,tp_host_runtime}.hpp`、
`include/strata/kernels/allreduce.hpp`、`include/strata/program/{tp_pair,tp_rules}.hpp`、
`src/kernels/cuda/allreduce.cu`，以及十个 `tests/core/tp_*` 与 `tests/program/tp_rules_test.cpp`）——
而本文只描述了我动过的那些部分。脚手架产物（崩溃转储、`.bak` 配置副本、本地 expert profile）未包含，
因此该分支不保证能独立构建。不要假设整个 diff 都是本文描述的内容。

## 8. 顺带找到并修掉的真 bug（即使其余被丢弃也值得保留）

1. **QSA 查询步长。** `split_q_kernel` 把 rank 的 query head 紧密排列
   （`q[i] = qf[t*H*512 + h*512 + d]`，`i = t*H*256 + h*256 + d`），而 `rope` / `rms_rows` / `gate_attn`
   都按 `qh*256` 寻址。两个注意力调用点却仍用模型的 `n_head`（24 → 6144）。已改为 rank 局部 `QsaShapes`
   加 `qstride = sq.n_head * head_dim`。这让 GDN 探针的非有限计数恰好移动了一个 token 的通道量，
   说明它是真缺陷，只是不是最后一个。
2. **专家缓存尺寸。** 重试按固定 3/4 缩小，并读取了上下文建立前的 `g_tp_free_bytes`，于是实际空闲
   13.29 GiB 时只分配了 11.46 GiB——白扔 1.83 GiB。现在按 `device_free_bytes()` 填充，这正是显存
   落在 19.44 GB 的原因。
3. **GDN 的 v-band 切分**（`shard.hpp`，在本基线里已存在）：连续对半切会破坏 `v % 16` 的
   v-head/q-head 配对，从而改变生成文本。

## 9. 我**没有**测的部分

- **只编了 CUDA。** `AGENTS.md` 要求任何改动涉及的文件都要编 CUDA + HIP + SYCL。这里的内核位于共享
  的 `.cu` 文件里，因此 HIP 与 SYCL 构建**未经验证**。
- **没有其它 GPU / 驱动。** 一台机器、一个 pack、一个模型。
- **没有 50K–100K 的 prompt 测量。** 正确路径约 40 tok/s，一个 50K prompt 单次约 20 分钟；我测到
  14 060 token。
- **没有质量评估。** 我只检查了 `12*7 → 84`、token 数是否完整读出、以及是否有非有限行。我**没有**跑
  召回/needle 套件，也没有和层切分做困惑度对比。
- **没有单卡逐位回归。** 所有新参数默认等于模型自身几何，且单卡时 `tp_ranks_current()` 为 1，
  所以意图是单卡无变化——但我没有逐位对比改动前后。
- 没测 `--spec 4` 以外更重的批量/投机配置，也没有长时间稳定性测试。

## 10. 先例：llama.cpp 与 ggml

本补丁的张量并行规则是照 llama.cpp 的 tensor mode 写的，Strata 已把它作为 pin 的依赖携带并在源码里
就地引用。除此之外没有别的来源主张。

- **许可：标准 MIT 全文** —— `third_party/llama.cpp/LICENSE` 为
  `MIT License / Copyright (c) 2023-2026 The ggml authors`，含标准授权段、"shall be included in all
  copies or substantial portions" 条件与 `AS IS` 免责。`third_party/ggml/LICENSE` 是同一份文本。
  两者都不是修改过或双许可的 MIT，也没有附加非商业、copyleft 或专利条款。
- **不随仓库分发**：`.gitignore:31` 忽略 `/third_party/llama.cpp`，`CMakeLists.txt` 用
  `STRATA_ORACLE_SOURCE_DIR` 指向用于 parity 测试的 pin 副本。pin 的树是下载的、不是提交的，
  因此 clone 本仓库根本不包含 llama.cpp。
- **取用了什么**：tensor mode 的规则，并按 Strata 自己的注释风格带 file:line 引用——
  `shard.hpp` 引 `handle_mul_mat:600-603`（按复制激活切专家轴）、`GGML_BACKEND_SPLIT_AXIS_PARTIAL:591-593`
  （哪些轴需要跨卡相加）、`MUL_MAT_ID` 的规则，以及 `qwen4exp.cpp:242-247` 的形状。
- **义务**：MIT 要求版权与许可声明随"实质性部分"一起传播。仓库的 `LICENSE` 已满足 Strata 自身的 MIT
  条款，就地 `file:line` 引用已标明出处。若 review 发现某函数是复制而非重写，请把 ggml 的声明
  放在它旁边。

## 11. 许可与免责

- Strata 为 **MIT**，Copyright (c) 2026 Niko1221 and the Strata contributors。任何再分发都必须保留
  `LICENSE` 及其版权与许可声明，不得改许可。
- MIT 文本使本软件**"AS IS"，不提供任何担保**。本文不构成任何适用性主张：快路径是坏的，正确路径
  比现有层切分更慢。
- 模型权重、native expert pack、expert profile 是**独立产物，不属于 Strata 的 MIT 源码**，
  不应提交进仓库。
- 本文不含任何凭据。

## 12. 我接下来会尝试的三步

1. 在 `kv_append_kernel` 里打印它对前 8 个 cell 写入 stage 的 `row_id`，同时在
   `prompt_attn_kernel`（`src/kernels/cuda/qsa_prompt_attn.cu` 的 `srow` 填充处）打印同 8 个 cell 算出的
   `r`，逐值对比。两者不一致即为根因。
2. dump `m.ident_table`（`prefill.cpp:1204` 构建它），确认 `ident[i] == i` 而不是别的排列。
3. 记住 `page_size == 4`，block 数是 `cells / 4`；任何按 64 推的结论都要重算。

## 13. 复现

```
cmake --build . --config Release --target strata
strata.exe --serve --tp 2 --pack <pack> --native <gguf> --ple-gguf <gguf> \
  --expert-profile <profile> --expert-cache auto --max-context 524288 --kv int8 \
  --kv-resident 32768 --prefill auto:16384 --pool-workers 11 --vision
# TP 必需：
STRATA_GDN_KEYHEAD=0
# 诊断：
STRATA_DBG_GDN=1        # GDN 与 QSA 块内逐步有限性检查（本补丁新增）
STRATA_DBG_NAN=1        # 原有；仅覆盖 MoE
STRATA_TP_WINDOWS=1     # 强制 verify 窗口 prompt 路径（正确的那条）
STRATA_TP_FASTPROMPT=1  # 批量快路径（当前超过一页即坏）
```
