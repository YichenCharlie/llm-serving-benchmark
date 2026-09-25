# vLLM Serving Project — 个人总结 Note

## Stage 1

配置相关的环境以及下载 vLLM 工具等。

## Stage 2

创建一个 baseline，作为后续实验的基准线。

## Stage 3

看 serving system 的并发能力。

并发数量提高，整体服务器吞吐 token 的速度提高。但并不是可以一直增长，超出服务器上限后，会饱和甚至恶化。

## Stage 4

修改 input 的长度，发现长度越长，TTFT 的时间就越长。

## Stage 5

中期总结，并给教授发套磁信。

## Stage 6

使用 Prefix Caching。

相同的 prefix 不需要重复计算 KV 数值，可以直接复用缓存中的计算，TTFT 速度变快。

## Stage 7

使用 Chunked Prefill。

规定 `max_num_batched_tokens`，让 Prefill 被 chunked，并穿插 Decode Work 中。

P99 的 ITL 会更好，但整体的 Mean ITL 会增加。

这是一个 trade-off，是一个取舍。

## Stage 8

Preemption：

假设 A、B、C、D 同时在 Running，可能会导致 KV Cache 不够，然后有一种“死锁”的感觉。

所以这时候主动释放 D，让 A、B、C 先继续，让 D 进入 Waiting，然后之后再 recompute。

---

## 总结

整个 LLM serving system 整体就是一个取舍。

为了让某些指标更好，可能进行的调整会让另一些指标下降。

---

## 我的个人 Question

### 为什么 Stage 8 不是单纯做 A/B 实验，打开 / 关闭 Preemption，然后看指标变化？

首先，vLLM V1 0.11.2 并没有给我们一个简单的：

```text
--enable-preemption
--disable-preemption
```

开关。

其次，我更应该关注的是：

> 当 KV Cache capacity 逐渐变紧时，vLLM 的 request residency、queueing 和 preemption 会怎样变化？

我很喜欢这句话：

> “完全禁用 preemption”还不一定是一个合理系统：如果所有 running requests 都缺 KV blocks、又不允许释放任何 request，系统有可能根本无法向前推进。Preemption 在这里不只是一个“性能优化开关”，它也是资源不足时保证系统继续工作的机制之一。

---

## 项目结束时GPT告诉我的一句话

> **gpt：你这个项目真正学到的，不只是怎么把某个指标调得更漂亮，而是开始理解一个系统为什么必须不断在吞吐、延迟、内存和调度之间做取舍。能看见这些 trade-off，并且知道怎么用实验把它们验证出来，这就是你从“会用工具”走向“理解系统”的那一步。**
