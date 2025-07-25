import requests
import wave
import pyaudio
import numpy as np
import resampy
import soundfile as sf
from scipy import signal
import time
import traceback

def apply_antialiasing(audio, original_rate, target_rate):
    """应用抗锯齿滤波器"""
    nyquist = 0.5 * min(original_rate, target_rate)
    cutoff = 0.9 * nyquist  # 保留90%的频带
    
    # 设计Butterworth低通滤波器
    b, a = signal.butter(4, cutoff/nyquist, 'low')
    return signal.filtfilt(b, a, audio)

def add_dither(audio_float):
    """添加TPDF抖动减少量化失真"""
    dither = np.random.random(size=audio_float.shape) * 2 - 1
    dither *= 0.5 / 32768  # 0.5 LSB的抖动幅度
    return audio_float + dither

def normalize_audio(audio_data):
    """改进的标准化方法，避免削波"""
    # 计算RMS能量
    rms = np.sqrt(np.mean(audio_data**2))
    
    # 动态范围压缩
    compression_factor = 1.0 / (rms * 3.0 + 0.1)
    compressed_audio = np.tanh(audio_data * compression_factor)
    
    # 转换为int16
    audio_int16 = (compressed_audio * 32767).astype(np.int16)
    
    # 计算并显示峰值和RMS
    peak = np.max(np.abs(audio_int16)) / 32767.0
    rms_db = 20 * np.log10(rms + 1e-10)
    print(f"音频标准化: Peak={peak:.2f}, RMS={rms_db:.1f} dB")
    
    return audio_int16

def find_best_device(p, target_rate=44100):
    """自动选择最佳播放设备（更健壮的版本）"""
    print("\n扫描音频输出设备...")
    best_device = None
    default_dev = p.get_default_output_device_info()
    
    print(f"默认输出设备: 索引 {default_dev['index']} - {default_dev['name']}")
    
    # 尝试获取设备列表
    try:
        device_count = p.get_device_count()
        print(f"发现 {device_count} 个音频设备")
        
        for i in range(device_count):
            try:
                dev = p.get_device_info_by_index(i)
                
                # 检查设备是否支持输出
                output_channels = dev.get('maxOutputChannels', 0)
                if output_channels > 0:
                    dev_name = dev.get('name', f"设备 {i}")
                    sample_rate = dev.get('defaultSampleRate', 0.0)
                    
                    print(f"设备 {i}: {dev_name} - 采样率: {sample_rate}Hz, 输出声道: {output_channels}")
                    
                    # 优先选择支持目标采样率的设备
                    if sample_rate == target_rate:
                        if best_device is None or output_channels > best_device.get('maxOutputChannels', 0):
                            best_device = dev
            except Exception as e:
                print(f"  设备 {i} 信息获取失败: {str(e)}")
    except Exception as e:
        print(f"设备扫描失败: {str(e)}")
        return default_dev['index']
    
    if best_device:
        print(f"✅ 选择最佳设备: {best_device['name']} (索引 {best_device['index']})")
        return best_device['index']
    
    print(f"⚠️ 未找到支持 {target_rate}Hz 的设备，使用默认输出设备")
    return default_dev['index']

def generate_tts_wav_and_play(json_data, server_host='10.16.80.147', server_port=50000):
    """生成TTS语音并播放，同时保存SIM900兼容版本"""
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
    print(f"生成TTS文本: {tts_text[:50]}... (共{len(tts_text)}字)")

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
    
    print(f"语音指令: {instruct_text}")

    # 发送请求 - 使用新的URL和POST方法
    url = f"http://{server_host}:{server_port}/generate_voice"
    print(f"正在向TTS服务器发送请求: {url}")
    
    start_time = time.time()
    try:
        response = requests.post(
            url,
            data={
                'text': tts_text,
                'instruct': instruct_text
            },
            stream=True,
            timeout=30  # 增加超时时间
        )
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请检查服务器状态")
        return
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {str(e)}")
        return
    
    request_time = time.time() - start_time
    print(f"请求完成 (耗时: {request_time:.2f}秒)")

    # 检查响应状态
    if response.status_code != 200:
        try:
            error_msg = response.json().get('error', '未知错误')
        except:
            error_msg = response.text[:100] + "..." if len(response.text) > 100 else response.text
        print(f"❌ 生成失败: 状态码 {response.status_code}, 错误: {error_msg}")
        return

    # 1. 从服务器接收完整的原始音频数据 (22050 Hz)
    original_rate = 22050
    raw_audio_data = response.content
    print(f"接收音频数据: {len(raw_audio_data)/1024:.1f} KB")
    
    # 保存原始音频用于调试
    with open('original_audio.wav', 'wb') as f:
        f.write(raw_audio_data)
    
    # 将原始字节数据转换为NumPy数组 (int16)，然后转为float32用于高质量处理
    audio_np_original = np.frombuffer(raw_audio_data, dtype=np.int16)
    audio_float_original = audio_np_original.astype(np.float32) / 32768.0
    
    # =================================================================
    # == 并行路径 1: 生成用于保存的 8kHz 音频 (SIM900兼容)
    # =================================================================
    sim900_rate = 8000
    print(f"生成SIM900兼容音频 (8kHz)...")
    
    # 从原始22.05kHz数据重采样到8kHz
    audio_resampled_8k_float = resampy.resample(
        audio_float_original, 
        original_rate, 
        sim900_rate, 
        filter='kaiser_best'
    )
    
    # 应用额外的低通滤波
    nyquist_8k = sim900_rate / 2
    b_8k, a_8k = signal.butter(4, 3400/nyquist_8k, 'low')
    audio_resampled_8k_float = signal.filtfilt(b_8k, a_8k, audio_resampled_8k_float)
    
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
    # == 并行路径 2: 生成用于播放的高质量音频
    # =================================================================
    # 使用44.1kHz代替48kHz以获得更好的整数倍转换
    PLAYBACK_RATE = 44100  # CD质量采样率
    
    print(f"🎧 准备高质量播放 (44.1kHz)...")
    
    # 添加抗锯齿滤波 (针对播放路径)
    print("应用抗锯齿滤波...")
    audio_float_play = apply_antialiasing(audio_float_original, original_rate, PLAYBACK_RATE)
    
    # 添加抖动处理
    print("添加抖动处理...")
    audio_float_play = add_dither(audio_float_play)
    
    # 使用resampy进行重采样（更可靠的方法）
    print("使用resampy进行高质量重采样...")
    audio_resampled_playback_float = resampy.resample(
        audio_float_play, 
        original_rate, 
        PLAYBACK_RATE, 
        filter='kaiser_best'
    )
    
    # 应用最终标准化
    audio_np_playback = normalize_audio(audio_resampled_playback_float)
    
    # 初始化PyAudio
    try:
        p = pyaudio.PyAudio()
        
        # 查找最佳播放设备
        # TARGET_DEVICE_INDEX = find_best_device(p, PLAYBACK_RATE)
        TARGET_DEVICE_INDEX = 18  # <-- 已手动指定设备
        print(f"✅ 已手动指定播放设备索引: {TARGET_DEVICE_INDEX}")
        
        # 尝试打开音频流
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=PLAYBACK_RATE,
                output=True,
                output_device_index=TARGET_DEVICE_INDEX,
                frames_per_buffer=4096  # 增加缓冲区大小
            )
        except Exception as e:
            print(f"❌ 打开音频流失败: {str(e)}")
            print("尝试使用默认设备...")
            # 尝试使用默认设备
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=PLAYBACK_RATE,
                output=True,
                frames_per_buffer=4096
            )
        
        # 分段播放，避免缓冲区溢出
        chunk_size = 4096
        total_frames = len(audio_np_playback)
        print(f"开始播放 (总帧数: {total_frames}, 时长: {total_frames/PLAYBACK_RATE:.2f}秒)")
        
        # 播放进度跟踪
        start_play_time = time.time()
        played_frames = 0
        
        try:
            for i in range(0, total_frames, chunk_size):
                end_index = min(i + chunk_size, total_frames)
                chunk = audio_np_playback[i:end_index].tobytes()
                stream.write(chunk)
                
                # 更新进度
                played_frames = end_index
                if played_frames % (PLAYBACK_RATE * 2) == 0:  # 每2秒更新一次
                    elapsed = time.time() - start_play_time
                    progress = min(100, played_frames / total_frames * 100)
                    print(f"播放进度: {progress:.1f}% ({elapsed:.1f}秒)")
        except Exception as e:
            print(f"❌ 播放过程中发生错误: {str(e)}")
            traceback.print_exc()
        
        # 计算实际播放时间
        play_time = time.time() - start_play_time
        print(f"✅ 语音播放完毕 (耗时: {play_time:.2f}秒)")
        
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"❌ PyAudio初始化失败: {str(e)}")
        traceback.print_exc()
    
    # 保存播放用的音频用于调试
    try:
        sf.write('playback_audio.wav', audio_np_playback, PLAYBACK_RATE)
        print("已保存播放音频用于调试: playback_audio.wav")
    except Exception as e:
        print(f"❌ 保存播放音频失败: {str(e)}")


# 示例用法（仅包含text、emo、voice字段）
if __name__ == "__main__":
    json_data = [
        {"text": "一年了？！", "emo": "担心", "voice": "提高"},
        {"text": "这公司也太欺负人了！", "emo": "愤怒", "voice": "有力"},
        {"text": "这样，您加我微信，把合同和欠薪证据拍给我，我先帮您做个免费分析？", "emo": "平静", "voice": "缓慢"}
    ]
    generate_tts_wav_and_play(json_data)
