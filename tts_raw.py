# -*- coding: utf-8 -*-
import sys
import re
import time
import wave
import pyaudio
import numpy as np
from typing import Dict, Any, Set, Callable, Optional, List
sys.path.append("..")

from common import credential
from tts import speech_synthesizer_ws
from common.log import logger
from common.utils import is_python3

APPID = 
SECRET_ID = 
SECRET_KEY = 

VOICETYPE = 101001  # 音色类型
FASTVOICETYPE = ""
CODEC = "pcm"  # 音频格式：pcm/mp3
SAMPLE_RATE = 16000  # 音频采样率：8000/16000
ENABLE_SUBTITLE = True


class MySpeechSynthesisListener(speech_synthesizer_ws.SpeechSynthesisListener):

    def __init__(self, id, codec, sample_rate):
        self.start_time = time.time()
        self.id = id
        self.codec = codec.lower()
        self.sample_rate = sample_rate

        self.audio_file = ""
        self.audio_data = bytes()

        # 初始化 pyaudio 播放器
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  output=True)

    def set_audio_file(self, filename):
        self.audio_file = filename

    def on_synthesis_start(self, session_id):
        super().on_synthesis_start(session_id)
        if not self.audio_file:
            self.audio_file = "speech_synthesis_output_" + str(self.id) + "." + self.codec
        self.audio_data = bytes()

    def on_audio_result(self, audio_bytes):
        super().on_audio_result(audio_bytes)
        self.audio_data += audio_bytes

        # 实时播放音频
        self.stream.write(audio_bytes)

    def on_synthesis_end(self):
        super().on_synthesis_end()
        # 关闭流和pyaudio
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        # # 保存为wav文件
        # if self.codec == "pcm":
        #     wav_fp = wave.open(self.audio_file + ".wav", "wb")
        #     wav_fp.setnchannels(1)
        #     wav_fp.setsampwidth(2)
        #     wav_fp.setframerate(self.sample_rate)
        #     wav_fp.writeframes(self.audio_data)
        #     wav_fp.close()
        # elif self.codec == "mp3":
        #     with open(self.audio_file, "wb") as fp:
        #         fp.write(self.audio_data)
        #
        # print(f"音频合成完成，文件已保存：{self.audio_file}")

    def on_synthesis_fail(self, response):
        super().on_synthesis_fail(response)
        err_code = response["code"]
        err_msg = response["message"]
        print(f"TTS合成失败，错误码：{err_code}，错误信息：{err_msg}")


def split_text_by_bg_sound(text):
    """
    拆分文本成文本和背景音标记序列
    例如：
    '今天天气真好($键盘)，($咳嗽)啦' ->
    ['今天天气真好', '键盘', '，', '咳嗽', '啦']
    """
    pattern = r'\(\$\s*(.*?)\s*\)'
    parts = []
    last_index = 0
    for m in re.finditer(pattern, text):
        if m.start() > last_index:
            parts.append(text[last_index:m.start()])
        parts.append(m.group(1))  # 背景音标记
        last_index = m.end()
    if last_index < len(text):
        parts.append(text[last_index:])
    return parts


def play_wav_file(wav_path, volume_factor=0.15):  # 默认降低到 30%
    try:
        wf = wave.open(wav_path, 'rb')
    except FileNotFoundError:
        print(f"背景音文件不存在：{wav_path}")
        return

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    chunk = 1024
    while True:
        data = wf.readframes(chunk)
        if not data:
            break

        # 调整音量
        audio_np = np.frombuffer(data, dtype=np.int16)
        audio_np = (audio_np * volume_factor).astype(np.int16)
        adjusted_data = audio_np.tobytes()

        stream.write(adjusted_data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


def process(id, text):
    logger.info(f"process start: idx={id} text={text}")

    parts = split_text_by_bg_sound(text)
    # parts 是文本和背景音标记的序列

    for idx, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if part in ("键盘", "翻书", "咳嗽"):
            # 播放对应背景音
            logger.info(f"play bg sound: {part}")
            print(f"播放背景音：{part}.wav")
            play_wav_file(f"{part}.wav")
        else:
            # 合成普通文本
            listener = MySpeechSynthesisListener(f"{id}_{idx}", CODEC, SAMPLE_RATE)
            credential_var = credential.Credential(SECRET_ID, SECRET_KEY)
            synthesizer = speech_synthesizer_ws.SpeechSynthesizer(
                APPID, credential_var, listener)
            synthesizer.set_text(part)
            synthesizer.set_voice_type(VOICETYPE)
            synthesizer.set_codec(CODEC)
            synthesizer.set_sample_rate(SAMPLE_RATE)
            synthesizer.set_enable_subtitle(ENABLE_SUBTITLE)
            synthesizer.set_fast_voice_type(FASTVOICETYPE)

            synthesizer.start()
            synthesizer.wait()
            logger.info(f"subprocess done: idx={id}_{idx} text={part}")

    logger.info(f"process done: idx={id} text={text}")
    return id

def read_tts_text():
    lines_list = []
    with open('test.txt', 'r', encoding='utf-8') as file:
        for line in file:
            lines_list.append(line.strip())
    return lines_list

#tts.py 最后添加
async def play_tts(call_id: str, text_data: List[Dict[str, Any]], voice_params: Optional[Dict[str, Any]] = None):
    """
    由服务端调用的播放函数
    text_data: [{"text": "你好($咳嗽)世界", "emo": "高兴"}, ...]
    """
    for idx, item in enumerate(text_data):
        text = item.get("text", "")
        print(f"[play_tts] 播放第{idx}段文本: {text}")
        process(f"{call_id}_{idx}", text)



if __name__ == "__main__":
    if not is_python3():
        print("only support python3")
        sys.exit(0)

    lines = read_tts_text()

    for idx, line in enumerate(lines):
        result = process(idx, line)
        print(f"\nTask {result} completed\n")
