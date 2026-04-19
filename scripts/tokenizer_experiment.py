"""
Day 2 - Tokenizer 实验:理解 Token 是什么
运行方式: python scripts/tokenizer_experiment.py
"""
from transformers import AutoTokenizer

MODEL_PATH = "./models/Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 6 组不同特性的文本
texts = [
    "南昌职业大学数智学院",
    "I love teaching AI.",
    "def hello(): print('world')",
    "🚀 人工智能 Artificial Intelligence",
    "在2026年,大模型变得无处不在",
    "ChatGPT、Claude、Gemini、Qwen 都是 LLM",
]

print(f"{'='*70}")
print(f"{'原文':<40} {'字符数':>6} {'Token 数':>8} {'比例':>6}")
print(f"{'='*70}")

for t in texts:
    tokens = tokenizer.tokenize(t)
    char_count = len(t)
    token_count = len(tokens)
    ratio = char_count / token_count if token_count > 0 else 0
    display_text = t if len(t) <= 35 else t[:32] + "..."
    print(f"{display_text:<40} {char_count:>6} {token_count:>8} {ratio:>6.2f}")

print(f"{'='*70}\n")

# 详细展示一个例子
print("🔍 详细展示:『南昌职业大学数智学院』的切分过程")
print("-" * 60)
t = "南昌职业大学数智学院"
tokens = tokenizer.tokenize(t)
ids = tokenizer.encode(t, add_special_tokens=False)

print(f"原文:       {t}")
print(f"切分 Token: {tokens}")
print(f"Token ID:   {ids}")
print(f"Token 数:   {len(tokens)}")
print()

# 反向验证
print("🔄 反向验证: 从 ID 还原回文字")
print(f"Decode 结果: {tokenizer.decode(ids)}")