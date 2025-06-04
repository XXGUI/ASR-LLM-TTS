这是一个结合了语音识别+大模型+语音合成的demo，基于https://github.com/ABexit/ASR-LLM-TTS进行修改
语音识别：SenseVoiceSmall
语音合成：vits
大模型：AutoModelForCausalLM或者外部api

文件说明：
main.py：监听麦克风+qwen2.5+tts对话（回车控制录音开始结束）
lingyin.py：实时监听麦克风对话
main-dify.py：监听麦克风+dify api接口推理+tts对话（回车控制录音开始结束）
main-dify-api.py：api对外形式的程序，提供了asr，tts对外接口，同时支持输入文本推理后返回音频结果，适合集成到实时对话业务。
