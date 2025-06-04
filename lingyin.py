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
from queue import Queue
import webrtcvad
import threading
import pyaudio
import wave
import re
from pypinyin import pinyin, Style
from modelscope.pipelines import pipeline

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
CHUNK = 1024              # 音频块大小
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
NO_SPEECH_THRESHOLD = 1   # 无效语音阈值，单位：秒
OUTPUT_DIR = "./output"   # 输出目录
folder_path = "./Test_QWen2_VL/"
audio_file_count = 0
audio_file_count_tmp = 0

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 队列用于音频和视频同步缓存
audio_queue = Queue()

# 全局变量
last_active_time = time.time()
recording_active = True
segments_to_save = []
saved_intervals = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间


# --- 唤醒词、声纹变量配置 ---
set_KWS = "ce shi"
flag_KWS = 0
flag_KWS_used = 1

flag_sv_used = 0
flag_sv_enroll = 0
thred_sv = 0.35

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

def system_introduction(text):
        global audio_file_count
        synthesize(text,model_path,config_path, pypinyin_local, folder_file,length_scale=0.8)
        play_audio(folder_file)
        
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

# 检测 VAD 活动
def check_vad_activity(audio_data):
    # 将音频数据分块检测
    num, rate = 0, 0.4
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False

# 音频录制线程
def audio_recorder():
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    audio_buffer = []
    print("音频录制已开始")
    
    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)
        
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 1.1:
            # 拼接音频数据并检测 VAD
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            if vad_result:
                print("检测到语音活动")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("静音中...")
            audio_buffer = []  # 清空缓冲区
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            # 检查是否需要保存
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                save_audio_video()
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
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

# 保存音频和视频
def save_audio_video():
    pygame.mixer.init()
    global segments_to_save,last_vad_end_time, saved_intervals
    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll

    if flag_sv_enroll:
        audio_output_path = f"{set_SV_enroll}/enroll_0.wav"
    else:
        audio_file_count += 1
        audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"
    if not segments_to_save:
        return
    # 停止当前播放的音频
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("检测到新的有效音，已停止当前音频播放")
    # 获取有效段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save]
    if flag_sv_enroll:
        audio_length = 0.5 * len(segments_to_save)
        if audio_length < 3:
            print("声纹注册语音需大于3秒，请重新注册")
            return 1
    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        system_introduction(text)
    else:
        inference_thread = threading.Thread(target=Inference, args=(audio_output_path,))
        inference_thread.start()
        # 记录保存的区间
        saved_intervals.append((start_time, end_time))
    # 清空缓冲区
    segments_to_save.clear()
    
# --- 播放音频 ---
def play_audio(file_path):
    try:
        if not pygame.mixer.get_init():
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

# --- 转拼音
def extract_chinese_and_convert_to_pinyin(input_string):
    # 使用正则表达式提取所有汉字
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', input_string)
    # 将汉字列表合并为字符串
    chinese_text = ''.join(chinese_characters)
    
    # 转换为拼音
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    # 将拼音列表拼接为字符串
    pinyin_text = ' '.join([item[0] for item in pinyin_result])
    
    return pinyin_text

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

set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
# -------- CAM++声纹识别 -- 模型加载 --------
sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

# --- 程序启动 ---
input_dev = check_input_device()
print(f"将使用录音设备: {input_dev}")

def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav"):
    global audio_file_count

    global set_SV_enroll
    global flag_sv_enroll
    global thred_sv
    global flag_sv_used

    global set_KWS
    global flag_KWS
    global flag_KWS_used
    
    os.makedirs(set_SV_enroll, exist_ok=True)
    # --- 如果开启声纹识别，且声纹文件夹为空，则开始声纹注册。设定注册语音有效长度需大于3秒
    if flag_sv_used and is_folder_empty(set_SV_enroll):
        text = f"无声纹注册文件！请先注册声纹，需大于三秒哦~"
        print(text)
        system_introduction(text)
        flag_sv_enroll = 1
    
    else:
        # -------- SenceVoice 推理 ---------
        input_file = (TEMP_AUDIO_FILE)
        res = model_senceVoice.generate(
            input=input_file,
            cache={},
            language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
        )
        prompt = res[0]['text'].split(">")[-1]
        prompt_pinyin = extract_chinese_and_convert_to_pinyin(prompt)
        print(prompt, prompt_pinyin)

        # --- 判断是否启动KWS
        if not flag_KWS_used:
            flag_KWS = 1
        if not flag_KWS:
            if set_KWS in prompt_pinyin:
                flag_KWS = 1
        
        # --- KWS成功，或不设置KWS
        if flag_KWS:
            sv_score = sv_pipeline([os.path.join(set_SV_enroll, "enroll_0.wav"), TEMP_AUDIO_FILE], thr=thred_sv)
            print(sv_score)
            sv_result = sv_score['text']
            if sv_result == "yes":
                result = send_chat_message(dify_app_key,prompt,"128机器测试",cache_conversation_id)
                text = result["answer"]
                cache_conversation_id = result["conversation_id"]
                synthesize(result,model_path,config_path, pypinyin_local, folder_file,length_scale=0.8)
                play_audio(folder_file)
            else:
                text = "很抱歉，你不是我的主人哦，我无法为您服务"
                system_introduction(text)
        else:
            text = "很抱歉，唤醒词错误，我无法为您服务。请说出正确的唤醒词哦"
            system_introduction(text)
            
if __name__ == "__main__":
    try:
        # 启动音视频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        audio_thread.start()

        flag_info = f'{flag_sv_used}-{flag_KWS_used}'
        dict_flag_info = {
            "1-1": "您已开启声纹识别和关键词唤醒，",
            "0-1":"您已开启关键词唤醒",
            "1-0":"您已开启声纹识别",
            "0-0":"",
        }
        if flag_sv_used or flag_KWS_used:
            text = dict_flag_info[flag_info]
            system_introduction(text)

        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        print("录制已停止")