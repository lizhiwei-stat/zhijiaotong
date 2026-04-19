"""
Day 2 - 下载 Qwen2.5-1.5B-Instruct 模型到本地
运行方式: python scripts/download_qwen.py
预计用时: 5-15 分钟(取决于网速),约 3GB 流量
"""
from modelscope import snapshot_download
import os

# 模型存放到项目的 models/ 目录下
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
CACHE_DIR = './models'

print(f"📥 开始下载模型: {MODEL_NAME}")
print(f"📁 存放目录: {os.path.abspath(CACHE_DIR)}")
print("=" * 60)

model_dir = snapshot_download(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

print("=" * 60)
print(f"✅ 下载完成!模型已保存到:\n   {model_dir}")