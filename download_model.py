import os
from modelscope.hub.snapshot_download import snapshot_download

def download_model(model_id, base_dir="./model"):
    target_path = os.path.join(base_dir, model_id.replace("/", "_"))
    if os.path.exists(target_path):
        print(f'模型 {model_id} 已存在，跳过下载。')
        return target_path

    os.makedirs(base_dir, exist_ok=True)
    print(f'开始下载模型：{model_id}')
    model_dir = snapshot_download(model_id, cache_dir=base_dir)
    
    # 创建一个软链接或复制到目标路径（可选）
    os.rename(model_dir, target_path)
    
    print(f'模型 {model_id} 下载完成，路径：{target_path}')
    return target_path

if __name__ == "__main__":
    models = [
        "iic/SenseVoiceSmall",
        "Qwen/Qwen2.5-1.5B-Instruct"
    ]
    for model_id in models:
        download_model(model_id)

    print("所有模型下载完成！")
