<div align="center">

  <a href="https://huggingface.co/sii-research/InnoMegrez2">
    <b>🤗 Hugging Face</b>
  </a> &nbsp;|&nbsp;
  <a href="./docs/tech_report.pdf">
    <b>📄 Tech Report</b>
  </a> &nbsp;|&nbsp;
  <a href="./assets/wechat-official.jpg">
    <b>💬 WeChat Official</b>
  </a> &nbsp;

  <br>

  <strong>中文 | [English](./README_EN.md)</strong>

</div>

# 更新日志

- [2025.09.15] 发布 [InnoMegrez2](https://github.com/sii-research/InnoMegrez2) 正式版本，模型结构和预览版本一致，训练数据总量从5T增加到8T，在各个测试集上表现更加均衡。
- [2025.07.24] 发布 [InnoMegrez2-Preview](https://github.com/sii-research/InnoMegrez2) 预览版本，专为终端设备设计的大模型，兼顾MoE的精度杠杆与Dense的总参数量友好。


# InnoMegrez2

## 模型简介

InnoMegrez2 是专为终端设备设计的大模型，兼顾MoE的精度杠杆与Dense的总参数量友好。本次发布的为Megrez 2.0正式版本，训练数据量8T Tokens，未来我们计划提升模型的推理和Agent能力。

## 基础信息

<div align="center">

| | |
|:---:|:---:|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 3x7B |
| **Activated Parameters** | 3B |
| **Experts Shared Frequency**| 3 |
| **Number of Layers** (Dense layer included) | 31 |
| **Number of Dense Layers** | 1 |
| **Attention Hidden Dimension** | 2048 |
| **MoE Hidden Dimension** (per Expert) | 1408 |
| **Number of Attention Heads** | 16 |
| **Number of Experts** | 64 |
| **Selected Experts per Token** | 6 |
| **Number of Shared Experts** | 4 |
| **Vocabulary Size** | 128,880 |
| **Context Length** | 32K |
| **Base Frequency of RoPE** | 1,000,000 |
| **Attention Mechanism** | GQA |
| **Activation Function** | SwiGLU |
</div>

## 性能测试

我们使用开源评测工具 [OpenCompass](https://github.com/open-compass/opencompass) 对 InnoMegrez2 进行了评测，部分评测结果如下表所示。

<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center">Metric</th>
<th align="center"><sup>InnoMegrez2<br></sup></th>
<th align="center"><sup>InnoMegrez2<br>-Preview</sup></th>
<th align="center"><sup>SmallThinker-21B<br>-A3B-Instruct</sup></th>
<th align="center"><sup>Qwen3-30B-A3B</sup></th>
<th align="center"><sup>Qwen3-8B</sup></th>
<th align="center"><sup>Qwen3-4B<br>-Instruct-2507</sup></th>
<th align="center"><sup>Phi4-14B<br>(nothink)</sup></th>
<th align="center"><sup>Gemma3-12B</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Activate Params (B)</td>
<td align="center"></td>
<td align="center">3.0</td>
<td align="center">3.0</td>
<td align="center">3.0</td>
<td align="center">3.3</td>
<td align="center">8.2</td>
<td align="center">4.0</td>
<td align="center">14.7</td>
<td align="center">12.2</td>
</tr>
<tr>
<td align="center">Stored Params (B)</td>
<td align="center"></td>
<td align="center">7.5</td>
<td align="center">7.5</td>
<td align="center">21.5</td>
<td align="center">30.5</td>
<td align="center">8.2</td>
<td align="center">4.0</td>
<td align="center">14.7</td>
<td align="center">12.2</td>
</tr>
<tr>
<td align="center">MMLU</td>
<td align="center">EM</td>
<td align="center">85.4</td>
<td align="center"><strong>87.5</strong></td>
<td align="center">84.4</td>
<td align="center">85.1</td>
<td align="center">81.8</td>
<td align="center">-</td>
<td align="center">84.6</td>
<td align="center">78.5</td>
</tr>
<tr>
<td align="center">GPQA</td>
<td align="center">EM</td>
<td align="center"><strong>58.8</strong></td>
<td align="center">28.8</td>
<td align="center">55.0</td>
<td align="center">44.4</td>
<td align="center">38.9</td>
<td align="center">62</td>
<td align="center">55.5</td>
<td align="center">34.9</td>
</tr>
<tr>
<td align="center">IFEval</td>
<td align="center">Inst<br>loose</td>
<td align="center"><strong>87.7</strong></td>
<td align="center">80.2</td>
<td align="center">85.8</td>
<td align="center">84.3</td>
<td align="center">83.9</td>
<td align="center">83.4</td>
<td align="center">63.2</td>
<td align="center">74.7</td>
</tr>
<tr>
<td align="center">MATH-500</td>
<td align="center">EM</td>
<td align="center"><strong>87.2</strong></td>
<td align="center">81.6</td>
<td align="center">82.4</td>
<td align="center">84.4</td>
<td align="center">81.6</td>
<td align="center">-</td>
<td align="center">80.2</td>
<td align="center">82.4</td>
</tr>
</tbody>
</table>
</div>

## 如何运行

### Transformers

推荐使用最新版本的 `transformers` 或者 `transformers>=4.52.4` 的版本。
以下是一个非常简单的代码片段示例，展示如何运行 InnoMegrez2 模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "sii-research/InnoMegrez2"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [
    {"role": "user", "content": "世界上最高的山峰是哪座？"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

model_outputs = model.generate(
    model_inputs,
    do_sample=True,
    max_new_tokens=1024
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)

# 世界上最高的山峰是珠穆朗玛峰（Mount Everest），位于喜马拉雅山脉的中尼边境。珠穆朗玛峰的海拔高度为8,848.86米（29,031.7英尺），这一数据是由中国和尼泊尔在2020年共同宣布的最新测量结果。珠穆朗玛峰不仅是登山爱好者的圣地，也是地理和科学研究的重要对象。
```

## 如何部署

### vLLM

需使用 `vllm>=0.10.1` 版本。在当前版本环境下，需对 vllm 相关文件进行一次补丁替换；后续我们将提交 pull request，尽早将该修改合并至 vllm 的正式版本中。

1. 找到你的vllm安装路径
```python
import vllm
print(vllm.__file__)
```

2. 替换vllm相关文件
```shell
cp ./demo/vllm/patch/layer.py <vllm_install_path>/model_executor/layers/fused_moe/
```

#### vLLM 离线
```shell
cd demo/vllm
export MODEL_PATH="sii-research/InnoMegrez2"
python3 infer_vllm_offline.py $MODEL_PATH
```
#### vLLM 在线
在终端中启动vLLM服务，命令如下
```shell
cd demo/vllm
export MODEL_PATH="sii-research/InnoMegrez2"
python3 serve_llm_online.py serve $MODEL_PATH --gpu-memory-utilization 0.9 --served-model-name megrez-moe --trust_remote_code
```

现在，可以通过curl发送请求
```shell
curl --location 'http://localhost:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-123456' \
--data '{
    "model": "megrez-moe",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "世界上最高的山峰是哪座？"
                }
            ]
        }
    ]
}'
```

### SGLang

推荐 `sglang>=0.4.9.post2` 的版本
```shell
cd demo/sglang
export MODEL_PATH="sii-research/InnoMegrez2" 
python3 infer_sglang_offline.py $MODEL_PATH
```


## 最佳实践

为了获得最佳性能，建议以下设置：

1. 采样参数：推荐使用 Temperature=0.7 和 TopP=0.9 。

2. 标准化输出格式：在基准测试时，我们建议使用提示来标准化模型输出，比如：
    * 数学问题：在提示中包含“请逐步推理，并将最终答案放在\boxed{}中。”
    * 选择题：在提示中添加以下 JSON 结构以标准化响应：“请在 answer 字段中仅以选择字母的形式显示您的选择，例如 "answer": "C" 。”

# 许可声明

我们所有的开源模型均采用Apache 2.0协议授权。


# 联系我们

如果您有任何问题，请随时提交GitHub issue或联系[微信群组](./assets/wechat-group.jpg)。