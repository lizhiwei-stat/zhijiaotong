"""
Day 2 - 第一次让 Qwen 说话
运行方式: python scripts/hello_qwen.py
CPU 推理预计耗时: 30-120 秒(取决于 CPU 性能)
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# ============ 1. 配置 ============
MODEL_PATH = "./models/Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cpu"  # 我们暂时用 CPU

print("🔧 正在加载 tokenizer 和模型...")
print(f"   模型路径: {MODEL_PATH}")
print(f"   运行设备: {DEVICE}")

t_start = time.time()

# ============ 2. 加载 Tokenizer ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ============ 3. 加载模型 ============
# torch_dtype=torch.float32 → CPU 上用 float32 兼容性最好
# 如果你后面装了 GPU,改成 torch.float16 可省一半显存
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map=DEVICE
)

t_load = time.time() - t_start
print(f"✅ 模型加载完成,用时 {t_load:.1f} 秒\n")

# ============ 4. 构造对话 ============
messages = [
    {"role": "system", "content": "你是一位资深的 Python 编程老师,回答简洁准确,善于用例子说明。"},
    {"role": "user", "content": "请批改这段 Python 代码,指出错误并给出正确版本:\n\ndef sum_list(lst):\n    total = 0\n    for i in range(len(lst) - 1):\n        total += lst[i]\n    return total"}
]


# 套用 Qwen 的 Chat Template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("=" * 60)
print("📝 模型实际看到的输入:")
print("=" * 60)
print(text)
print("=" * 60)

# ============ 5. 编码 → 生成 → 解码 ============
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
print(f"\n📊 输入 token 数: {inputs.input_ids.shape[1]}")
print("\n🤔 模型思考中...(CPU 推理需要几十秒,请耐心等待)\n")

t_gen_start = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=256,      # 最多生成 256 个新 token
    temperature=0.7,          # 生成的随机性(0=确定,1=随机)
    do_sample=True,           # 开启采样(配合 temperature)
    top_p=0.9                 # 核采样
)

t_gen = time.time() - t_gen_start

# 只取新生成的部分(去掉输入)
response_ids = outputs[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(response_ids, skip_special_tokens=True)

n_new_tokens = len(response_ids)
speed = n_new_tokens / t_gen

# ============ 6. 显示结果 ============
print("=" * 60)
print("🤖 Qwen 的回答:")
print("=" * 60)
print(response)
print("=" * 60)
print(f"\n⏱️ 生成用时: {t_gen:.1f} 秒")
print(f"📈 生成 {n_new_tokens} 个 token,速度 {speed:.2f} token/秒")