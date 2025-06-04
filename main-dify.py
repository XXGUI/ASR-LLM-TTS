from vits_chinese.tts_module import synthesize
from funasr import AutoModel
import torchaudio
import pygame
import time
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import shutil
import requests
import json

# --- 检查输入设备 ---
def check_input_device():
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    if not input_devices:
        print("没有可用的录音输入设备！程序退出。")
        sys.exit(1)
    print("可用录音设备列表:")
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']}")
    selected_device = input_devices[0]['name']
    print(f"默认选择第一个录音设备: {selected_device}")
    return selected_device

# --- 录音 ---
def record_audio(filename="output.wav", sample_rate=44100, device=None):
    print("按下 Enter 开始录音...")
    input()
    print("录音中... 按下 Enter 结束录音")
    recording = []
    try:
        def callback(indata, frames, time, status):
            recording.append(indata.copy())
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, device=device):
            input()
    except Exception as e:
        print(f"录音错误: {e}")
        return
    audio_data = np.concatenate(recording, axis=0)
    write(filename, sample_rate, (audio_data * 32767).astype(np.int16))
    print(f"录音保存为 {filename}")

# --- 播放音频 ---
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

# --- 清空文件夹 ---
def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 不存在，已创建")
        return
    items = os.listdir(folder_path)
    if not items:
        print(f"文件夹 '{folder_path}' 已经为空")
        return
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

# --- 调用大模型 ---
cache_conversation_id = ""
def send_chat_message(api_key, query, user_id, conversation_id):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": conversation_id,
        "user": user_id
    }
    response = requests.post("http://127.0.0.1:81/v1/chat-messages", json=payload, headers=headers, stream=True)
    if response.status_code == 200:
        response_json = response.json()
        cache_conversation_id = response_json.get("conversation_id", conversation_id)
        answer = response_json.get("answer", "")
        new_conversation_id = response_json.get("conversation_id", conversation_id)
        return {
            "answer": answer,
            "conversation_id": new_conversation_id
        }
    else:
        return {
            "answer": "",
            "conversation_id": "",
            "status_code": response.status_code,
            "error": response.text
        }
        
# ------------------- 模型初始化 -------------------
model_dir = r"./model/iic_SenseVoiceSmall"
model_senceVoice = AutoModel(model=model_dir, trust_remote_code=True)

# 初始化模型，修改为你自己模型和配置路径
model_path = "/home/ASR-LLM-TTS-master/model/tts_models/G_AISHELL.pth"
config_path = "/home/ASR-LLM-TTS-master/model/tts_models/config.json"
pypinyin_local = "vits_chinese/misc/pypinyin-local.dict"

input_file = "./input_wav/input_wav.wav"
folder_file = "./out_answer/out_wav_0.wav"

dify_app_key = "app-qQNNBrbeLP3IqglcB0HS6gUC"

# --- 程序启动 ---
input_dev = check_input_device()
print(f"将使用录音设备: {input_dev}")

while True:
    times = {}
    # --- 录音 ---
    record_audio(input_file, device=input_dev)
    # --- 语音识别 ---
    print("语音识别中...")
    t1 = time.time()
    res = model_senceVoice.generate(
        input=input_file,
        cache={},
        language="auto",
        use_itn=False,
    )
    times["语音识别耗时"] = time.time() - t1

    # --- 构建 Prompt 并生成文本回复 ---
    print("大模型推理中...")
    t2 = time.time()
    prompt = res[0]['text'].split(">")[-1]
    print("输入:", prompt)
    result = send_chat_message(dify_app_key,prompt,"128机器测试",cache_conversation_id)
    response = result["answer"]
    cache_conversation_id = result["conversation_id"]
    times["模型推理耗时"] = time.time() - t2
    print("输出:", response)
    
    # --- 检查 response 类型 ---
    if not isinstance(response, str):
        print("❌ 无效响应，跳过语音合成。错误信息：", response)
        continue

    # --- 清理文件夹 ---
    clear_folder('out_answer')
    # --- 文本转语音合成 ---
    print("语音合成中...")
    t4 = time.time()
    synthesize(response,model_path,config_path, pypinyin_local, folder_file,length_scale=0.8)
    times["语音合成耗时"] = time.time() - t4
    # --- 打印总耗时统计 ---
    print("\n---- 本轮处理耗时统计 ----")
    total = sum(times.values())
    for stage, duration in times.items():
	    print(f"{stage}: {duration:.2f} 秒")
    print("--------------------------")
    print(f"总耗时: {total:.2f} 秒\n")
    # --- 播放合成语音 ---
    play_audio(folder_file)