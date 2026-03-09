# Qwen3-ASR 文档索引

## 使用指南 (guides/)

| 文档 | 说明 |
|------|------|
| [deployment_reference.md](guides/deployment_reference.md) | **部署快速参考** — 所有部署模式的命令速查（原生 / Docker / 各后端） |
| [usage.md](guides/usage.md) | 安装、启动服务、API 调用的完整使用指南 |
| [vllm_deployment_notes.md](guides/vllm_deployment_notes.md) | vLLM 版本锁定、音频依赖、Docker 部署注意事项 |
| [vllm_language_usage.md](guides/vllm_language_usage.md) | 如何在 vLLM 服务中指定转录语言 |

## 源码与架构分析 (architecture/)

| 文档 | 说明 |
|------|------|
| [streaming_code_walkthrough.md](architecture/streaming_code_walkthrough.md) | `qwen-asr-demo-streaming` 源码逐段解析 |
| [streaming_demo_analysis.md](architecture/streaming_demo_analysis.md) | `qwen-asr-demo-streaming` 源码深度解析 |
| [serve_vs_demo_diff_analysis.md](architecture/serve_vs_demo_diff_analysis.md) | `qwen-asr-serve` 与 `qwen-asr-demo` 结果差异分析 |

## 设计方案 (design/)

| 文档 | 说明 |
|------|------|
| [unified_service_plan.md](design/unified_service_plan.md) | 统一推理服务开发计划 |
