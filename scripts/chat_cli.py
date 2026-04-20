"""
Day 3 - 命令行多轮对话
运行方式: python scripts/chat_cli.py

特色:
- 支持多轮对话记忆
- 实时显示推理速度
- 输入 /clear 清空历史
- 输入 /exit 退出
- 输入 /save 保存当前对话到 docs/
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os
from datetime import datetime

MODEL_PATH = "./models/Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "你是职教通 AI 助手,由李志伟开发,专注于帮助职业院校师生的教学工作。回答要简洁、准确、友好。"

def load_model():
    """加载模型"""
    print("🔧 正在加载 Qwen2.5-1.5B-Instruct,首次加载约 30 秒...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("✅ 模型加载完成\n")
    return tokenizer, model

def chat_once(tokenizer, model, history):
    """单轮对话:接收用户输入,生成回复,更新历史"""
    user_input = input("\n👤 你: ").strip()

    # 处理特殊命令
    if user_input == "/exit":
        return "EXIT", history
    if user_input == "/clear":
        print("🧹 对话历史已清空\n")
        return "CONTINUE", []
    if user_input == "/save":
        save_history(history)
        return "CONTINUE", history
    if not user_input:
        return "CONTINUE", history

    # 组装 messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    # 生成回复
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("\n🤖 Qwen 思考中...", end="", flush=True)
    t_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    t_gen = time.time() - t_start

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    n_tokens = len(response_ids)

    print(f"\r🤖 Qwen: {response}")
    print(f"\n   [⏱️ {t_gen:.1f}s | 📝 {n_tokens} tokens | ⚡ {n_tokens/t_gen:.2f} tok/s]")

    history.append((user_input, response))
    return "CONTINUE", history

def save_history(history):
    """保存对话历史到 Markdown"""
    if not history:
        print("⚠️ 对话历史为空,没什么可保存的")
        return

    os.makedirs("docs/chats", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"docs/chats/chat_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 对话记录 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"> 模型: Qwen2.5-1.5B-Instruct\n\n")
        for i, (u, a) in enumerate(history, 1):
            f.write(f"## 第 {i} 轮\n\n")
            f.write(f"**👤 用户**:\n\n{u}\n\n")
            f.write(f"**🤖 助手**:\n\n{a}\n\n")
            f.write("---\n\n")

    print(f"✅ 对话已保存到 {filename}")

def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("🎓 职教通 AI 助手 · 命令行版")
    print("=" * 60)
    print("💡 可用命令:")
    print("   /clear  清空对话历史")
    print("   /save   保存当前对话到文件")
    print("   /exit   退出程序")
    print("=" * 60)

def main():
    print_welcome()
    tokenizer, model = load_model()
    history = []

    while True:
        status, history = chat_once(tokenizer, model, history)
        if status == "EXIT":
            # 退出前询问是否保存
            if history:
                save = input("\n💾 是否保存本次对话? (y/N): ").strip().lower()
                if save == "y":
                    save_history(history)
            print("\n👋 再见!")
            break

if __name__ == "__main__":
    main()