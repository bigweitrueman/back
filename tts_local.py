import requests
import wave
import pyaudio
import numpy as np
import resampy

def normalize_audio(audio_data):
    """将音频数据标准化到-1.0到1.0之间，并转换为int16格式"""
    # 找到浮点音频数据的最大绝对值
    max_val = np.max(np.abs(audio_data))
    # 如果最大值为0（静音），则无需处理
    if max_val > 0:
        # 将所有值除以最大绝对值，使范围缩放到 [-1.0, 1.0]
        normalized_audio = audio_data / max_val
    else:
        normalized_audio = audio_data
    
    # 将范围从 [-1.0, 1.0] 映射到 int16 的范围 [-32767, 32767]
    audio_int16 = (normalized_audio * 32767).astype(np.int16)
    return audio_int16

def generate_tts_wav_and_play(json_data, server_host='10.16.80.147', server_port=50000):
    # 仅JSON数据（只处理text、emo、voice）
    tts_segments = []
    first_emo = ""
    first_voice = ""
    found_first = False  # 标记是否已获取第一个emo和voice

    for item in json_data:
        # 只保留text字段处理
        if 'text' in item:
            tts_segments.append(item['text'])
            # 只获取第一个text项的emo和voice
            if not found_first:
                first_emo = item.get('emo', '')
                first_voice = item.get('voice', '')
                found_first = True  # 已获取，后续不再处理

    # 拼接完整文本
    tts_text = ''.join(tts_segments)

    # 构建中文指令文本（仅使用第一个emo和voice）
    instruct_parts = []
    if first_emo:
        instruct_parts.append(f"{first_emo}的语气")
    if first_voice:
        instruct_parts.append(f"{first_voice}的语调")

    if instruct_parts:
        instruct_text = f"用{', '.join(instruct_parts)}朗读这段文字"
    else:
        instruct_text = "用自然的语气朗读这段文字"

    # 发送请求 - 使用新的URL和POST方法
    url = f"http://{server_host}:{server_port}/generate_voice"
    print(f"正在向TTS服务器发送请求: {url}")
    response = requests.post(
        url,
        data={
            'text': tts_text,
            'instruct': instruct_text
        },
        stream=True
    )

    # 检查响应状态
    if response.status_code != 200:
        print("生成失败:", response.json())
        return

    # --- 最终优化：并行处理，避免二次采样以保证最高音质 ---

    # 1. 从服务器接收完整的原始音频数据 (22050 Hz)
    original_rate = 22050
    raw_audio_data = response.content
    
    # 将原始字节数据转换为NumPy数组 (int16)，然后转为float32用于高质量处理
    audio_np_original = np.frombuffer(raw_audio_data, dtype=np.int16)
    audio_float_original = audio_np_original.astype(np.float32)

    # =================================================================
    # == 并行路径 1: 生成用于保存的 8kHz 音频
    # =================================================================
    sim900_rate = 8000
    # 从原始22.05kHz数据重采样到8kHz
    audio_resampled_8k_float = resampy.resample(
        audio_float_original, 
        original_rate, 
        sim900_rate, 
        filter='kaiser_best'
    )
    # 标准化并转换回int16用于保存
    audio_np_8k_for_saving = normalize_audio(audio_resampled_8k_float)

    # 保存为 SIM900 兼容的 8kHz WAV 文件
    output_file_8k = 'generated_voice_8k.wav'
    with wave.open(output_file_8k, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sim900_rate)
        wf.writeframes(audio_np_8k_for_saving.tobytes())
    print(f"✅ 已保存SIM900兼容文件: {output_file_8k} (8kHz, 16-bit, Mono)")

    # =================================================================
    # == 并行路径 2: 生成用于播放的 48kHz 音频
    # =================================================================
    TARGET_DEVICE_INDEX = 22
    PLAYBACK_RATE = 48000
    
    print(f"🎧 准备从原始音频重采样至 {PLAYBACK_RATE}Hz 并通过设备 {TARGET_DEVICE_INDEX} 播放...")

    # 从原始22.05kHz数据重采样到48kHz
    audio_resampled_playback_float = resampy.resample(
        audio_float_original, 
        original_rate, 
        PLAYBACK_RATE, 
        filter='kaiser_best' # 使用最高质量的算法
    )
    # 标准化并转回int16用于播放
    audio_np_playback = normalize_audio(audio_resampled_playback_float)

    # 播放重采样后的音频
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=PLAYBACK_RATE,
        output=True,
        output_device_index=TARGET_DEVICE_INDEX
    )
    
    stream.write(audio_np_playback.tobytes())

    # 清理资源
    stream.stop_stream()
    stream.close()
    p.terminate()

    print(f"✅ 语音播放完毕。")


# # 示例用法（仅包含text、emo、voice字段）
# json_data = [
#     {"text": "一年了？！", "emo": "担心", "voice": "提高"},
#     {"text": "这公司也太欺负人了！", "emo": "愤怒", "voice": "有力"},
#     {"text": "这样，您加我微信，把合同和欠薪证据拍给我，我先帮您做个免费分析？", "emo": "平静", "voice": "缓慢"}
# ]
# generate_tts_wav_and_play(json_data)