# Qwen3-ASR v1.2.0 发布记录

时区：Asia/Shanghai。镜像身份以不可变 Image ID 为准，不以可变 tag 为准。

## 发布前基线（2026-08-08）

| 服务 | 旧镜像 | Image ID | 状态 |
| --- | --- | --- | --- |
| vLLM gateway + engine（6001） | `mygpt-qwen3-asr:v1.1.0` | `sha256:7b2a549638defe4d25b505e5f9846b49514dabb4bd248807367f8c594c4567ce` | running |
| Transformers ASR + gateway（6002） | `mygpt-qwen3-asr-hotword:v1.1.0` | `sha256:0e9ca8181e31d5361f6ce0c864350323916069edd059f42ee44d42324696b99c` | running |
| 业务前端 | `mygpt-audio-streaming-frontend:v1.8` | `sha256:c3897116e4d02b15afacfed64377ae8be8a9f2b4e12741b8b258d825329b514a` | running |

## 目标镜像

| 服务 | 目标 tag | Image ID | 部署结果 |
| --- | --- | --- | --- |
| vLLM gateway + engine | `mygpt-qwen3-asr:v1.2.0` | `sha256:022890f8594c36f31bd0416e159c004dbb3955d8c496d40d840e4f2a2d818b0e` | 已部署到 6001，healthy |
| Transformers ASR + gateway | `mygpt-qwen3-asr-hotword:v1.2.0` | `sha256:236ff7fb31ef04ef8749183528bcd1d6cd3709e60f14f985d6870ffd7792ceb1` | 已部署到 6002，healthy |
| 业务前端 | `mygpt-audio-streaming-frontend:v1.9` | 未生成 | PyArmor trial 拒绝部分文件，发布在 build 阶段停止；v1.8 继续运行 |

## 发布门槛与现场结果

- 服务端单元/契约测试：`227 passed`。
- 6001 HTTP：无上下文、数组热词、prompt+数组热词均返回合法 `text`；MP3 无上下文也走共享解码路径。
- 6001 SSE：`started -> partial* -> segment_final* -> final`、严格递增 sequence、毫秒时间戳和 `final:true` 均通过。
- 6001 WS：数组热词及 `Vocabulary:` 字符串热词均通过；ready sequence 0，后续 sequence、时间戳和 final 均通过。
- 6002 HTTP / SSE / WS：使用同一矩阵通过；数组及字符串热词均通过。
- 6001/6002 无语音 SSE：都只返回 `started -> final`，无 `segment_final` 且最终文本为空。
- 前端 Python：非 integration 全量测试通过（7963 selected）；JavaScript：143 suites / 5547 tests 通过。
- Qwen HTTP-only、SenseVoice fake、SSE 故障矩阵、分块全有或全无、staging 批次原子性和 realtime 降级均已通过自动化契约/故障注入测试。
- 6001/6002 的 60 秒与 239.329 秒整文件 SSE 现场验证通过；两端分别得到 3 段与 11 段，sequence 和时间范围均单调、不重叠且未越界。
- 生产前端导入、播放器和 GridFS 端到端：因 v1.9 未通过发布构建门槛，未在生产容器执行；不得把协议现场验证或自动化测试登记成前端生产验收。

## 前端发布阻塞（2026-08-08 15:28 CST）

`make deploy DEPLOY_VERSION=v1.9` 在 PyArmor trial 阶段对多个文件持续返回
`out of license`。`PYARMOR_ALLOW_PARTIAL` 保持关闭，没有生成或部署半加密镜像。
发布命令在第三个受限文件重试时主动停止，生产容器始终运行原 v1.8 Image ID
`sha256:c3897116e4d02b15afacfed64377ae8be8a9f2b4e12741b8b258d825329b514a`。

为隔离“应用/Docker 构建错误”和“加密授权错误”，额外构建了未部署的本地验证镜像
`mygpt-audio-streaming-frontend:v1.9-validation-unencrypted`
（当前 Image ID `sha256:f2db71d80e21a6a6daf67272c09418abf51f07924c5fddd275d84df15d29f446`）。
该镜像验证了 canonical Qwen 配置、readiness capability、旧类型别名、SenseVoice
HTTP-only 能力以及旧连接字段拒绝和字段值脱敏。它不是发布镜像，也未替换生产容器。
随后用禁用 Mongo 的临时容器分别只切换 URL/MODEL 连接 6001 与 6002；两次
`/health` 都报告 `asr=healthy`、`asr_type=qwen3`、`asr_service_name=Qwen3ASR`。
`/api/config/features` 只暴露业务 feature，热词和 prompt 均启用，没有泄露
backend/model/server URL。两个临时容器验收后已停止并自动删除。

最终对抗性复审额外验证了全部 segment/GridFS 读取路径，并修复两项发布边界：
GridFS 直接下载不能通过 metadata fallback 暴露未发布批次；Mongo 发布指针更新结果
不确定时不再补偿删除可能已经发布的数据。未引用 staging 批次仍受 24 小时安全窗
保护，并已纳入每 5 分钟周期清理，避免“重启时尚未过期”后永久残留。

继续发布需要满足以下任一条件：

1. 配置可覆盖当前源码规模的正式 PyArmor license 后，重新执行原发布命令；
2. 由发布负责人明确批准不加密构建；
3. 由发布负责人明确批准部分明文构建。

默认建议选 1；2/3 会改变既有源码保护边界，不应静默采用。

### 授权状态复核（2026-08-08 16:03 CST）

使用独立探针 tag 再次执行全量加密构建，参数保持
`ENABLE_PYARMOR_ENCRYPTION=true`、`PYARMOR_ALLOW_PARTIAL=`，并将授权错误设置为
首次出现即熔断。builder 明确报告 `Pyarmor 9.2.1 (trial), 000000, non-profits`，
随后对 48,288 字节的 `mission/i18n/strings-extractor.py` 返回
`ERROR out of license`。验证镜像中的 `pyarmor --version` 同时明确注明 trial license
不能混淆大脚本；当前运行时代码另有 16 个超过 32 KiB 的 Python 文件，因此排除
该非运行时脚本也不能满足完整加密发布门槛。探针未生成目标镜像、未覆盖正式 tag、
未重启生产容器。

## 回滚

- 6001 previous：发布前 Image ID `sha256:7b2a549638defe4d25b505e5f9846b49514dabb4bd248807367f8c594c4567ce`。
- 6002 previous：发布前 Image ID `sha256:0e9ca8181e31d5361f6ce0c864350323916069edd059f42ee44d42324696b99c`。
- 前端 previous：发布前 Image ID `sha256:c3897116e4d02b15afacfed64377ae8be8a9f2b4e12741b8b258d825329b514a`。
- 新前端遇到缺少 `capabilities` 的旧 ASR 时按 HTTP-only 运行，因此回滚 ASR 不要求同步回滚前端。
