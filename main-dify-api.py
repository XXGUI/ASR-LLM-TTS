from flask import Flask, request, jsonify
from vits_chinese.tts_module import synthesize
from funasr import AutoModel
import time, os
import requests
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
# 日志文件路径
LOG_FILE = "./dify-api.log"
# 设置 logger
logger = logging.getLogger("chat_logger")
logger.setLevel(logging.INFO)
# 文件日志处理器（最大5MB，保留3个旧日志）
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
# 添加到 logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 初始化模型
model_senceVoice = AutoModel(model="./model/iic_SenseVoiceSmall", trust_remote_code=True)
model_path = "./model/tts_models/G_AISHELL.pth"
config_path = "./model/tts_models/config.json"
pypinyin_local = "vits_chinese/misc/pypinyin-local.dict"

app = Flask(__name__)
cache_conversation_id = ""

UPLOAD_FOLDER = "./input_wav"
OUTPUT_FOLDER = "./out_answer"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------- 调用大模型 -------------------
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
        return {
            "answer": response_json.get("answer", ""),
            "conversation_id": response_json.get("conversation_id", conversation_id)
        }
    else:
        return {
            "answer": "",
            "conversation_id": "",
            "status_code": response.status_code,
            "error": response.text
        }

# ------------------- API 接口 -------------------

# 语音识别接口
@app.route("/asr", methods=["POST"])
def asr():
    audio = request.files.get("file")
    if not audio:
        return jsonify({"code": 400, "msg": "未提供音频文件", "data": None}), 400

    filename = secure_filename(audio.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(input_path)
    t1 = time.time()
    try:
        res = model_senceVoice.generate(
            input=input_path,
            cache={},
            language="auto",
            use_itn=False,
        )
        duration = time.time() - t1
        text = res[0]['text'].split(">")[-1]
        logger.info(f"[ASR] 输入文件: {input_path}")
        logger.info(f"[ASR] 输出文本: {text}")
        logger.info(f"[ASR] 耗时: {duration:.2f} 秒")
        return jsonify({
            "code": 200,
            "msg": None,
            "data": text
        })
    except Exception as e:
        logger.error(f"[ASR] 识别失败: {e}")
        return jsonify({"code": 500, "msg": "语音识别出错", "data": None}), 500


# 文本转语音接口
@app.route("/tts", methods=["POST"])
def tts():
    text = request.form.get("text")
    if not text:
        return jsonify({"code": 400, "msg": "未提供文本", "data": None}), 400
    output_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, f"chat_{int(time.time())}.wav"))
    t1 = time.time()
    try:
        synthesize(text, model_path, config_path, pypinyin_local, output_path)
        duration = time.time() - t1
        logger.info(f"[TTS] 输入文本: {text}")
        logger.info(f"[TTS] 输出音频: {output_path}")
        logger.info(f"[TTS] 耗时: {duration:.2f} 秒")
        return jsonify({
            "code": 200,
            "msg": None,
            "data": output_path
        })
    except Exception as e:
        logger.error(f"[TTS] 合成失败: {e}")
        return jsonify({"code": 500, "msg": "语音合成出错", "data": None}), 500

# 对话接口：语音识别 -> 大模型 -> 语音合成
conversation_cache = {}  # {phone: conversation_id}
@app.route("/chat", methods=["POST"])
def chat():
    phone = request.form.get("phone")
    audio_path = request.form.get("audioPath")
    query = request.form.get("query")
    if not phone:
        return jsonify({"code": 400, "msg": "未提供手机号", "data": None}), 400
    if not audio_path and not query:
        return jsonify({"code": 400, "msg": "audioPath和 query 不能同时为空", "data": None}), 400
    times = {}
    prompt = ""
    # --- ASR 语音识别 ---
    if audio_path:
        if not os.path.exists(audio_path):
            return jsonify({"code": 400, "msg": "音频路径无效或文件不存在", "data": None}), 400
        t1 = time.time()
        res = model_senceVoice.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=False,
        )
        times["ASR语音识别"] = time.time() - t1
        prompt = res[0]['text'].split(">")[-1]
    else:
        prompt = query
    # --- 对话生成 ---
    t2 = time.time()
    current_conversation_id = conversation_cache.get(phone, "")
    result = send_chat_message("app-8RhTqEmpf9q2XKTDhlC0IzM3", prompt, phone, current_conversation_id)
    response = result["answer"]
    conversation_cache[phone] = result["conversation_id"]
    times["DIFY大模型"] = time.time() - t2
    if not isinstance(response, str) or not response.strip():
        return jsonify({
            "code": 500,
            "msg": "大模型无有效响应",
            "data": None
        }), 500

    # --- 文本转语音 ---
    t3 = time.time()
    output_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, f"chat_{int(time.time())}.wav"))
    synthesize(response, model_path, config_path, pypinyin_local, output_path)
    times["TTS语音合成"] = time.time() - t3
    # --- 打印日志 ---
    logger.info("==== Chat Log ====")
    logger.info(f"[CHAT]手机号: {phone}")
    logger.info(f"[CHAT]输入文本: {prompt}")
    logger.info(f"[CHAT]模型回复: {response}")
    logger.info(f"[CHAT]输出音频: {output_path}")
    logger.info("[CHAT]耗时统计:")
    for k, v in times.items():
        logger.info(f"  {k}: {v:.2f} 秒")
    logger.info(f"[CHAT]总耗时: {sum(times.values()):.2f} 秒")
    logger.info("==================\n")
    return jsonify({
        "code": 200,
        "msg": None,
        "data": output_path
    })

# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
