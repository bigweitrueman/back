import asyncio
import websockets
import json
import argparse
from typing import Dict, Any, Set, Callable, Optional, List
from abc import ABC, abstractmethod
from tts_local import generate_tts_wav_and_play

parser = argparse.ArgumentParser(description="Phone Module Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
parser.add_argument("--port", type=int, default=8765, help="Port to bind")
parser.add_argument("--version", type=str, default="1.1", help="Protocol version")
parser.add_argument("--audio_device", type=str, help="Audio device name")
args = parser.parse_args()

class DeviceDriver(ABC):
    def __init__(self):
        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.version = "1.1"
        self.call_states: Dict[str, Dict[str, Any]] = {}  # call_id -> 状态

    @abstractmethod
    async def dial(self, phone_number: str, call_id: str = "") -> bool:
        """
        拨号接口
        Args:
            phone_number: 电话号码
            call_id: 通话ID（可选）
        Returns:
            bool: 拨号指令是否成功发送
        """
        pass

    @abstractmethod
    async def hangup(self, call_id: str) -> bool:
        """
        挂断接口
        Args:
            call_id: 通话ID
        Returns:
            bool: 挂断指令是否成功发送
        """
        pass

    @abstractmethod
    async def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """
        获取通话状态
        Args:
            call_id: 通话ID
        Returns:
            Dict: 状态信息 {"status": "connected", "duration": 30, ...}
        """
        pass

    @abstractmethod
    async def start_hardware_monitoring(self):
        """
        启动硬件事件监听
        持续监听硬件状态变化，通过event_callback推送事件
        """
        pass

    async def run(self):
        """启动硬件驱动"""
        await self.start_hardware_monitoring()

class ASRDriver(ABC):
    def __init__(self):
        self.asr_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.version = "1.1"
        self.active_calls: Dict[str, Dict[str, Any]] = {}  # call_id -> 状态

    @abstractmethod
    async def send_tts(self, call_id: str, text_data: List[Dict[str, Any]], voice_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        发送TTS文本进行语音合成和播放
        Args:
            call_id: 通话ID
            text_data: 文本数据数组 [{"text": "...", "emo": "...", "voice": "..."}, ...]
            voice_params: 语音参数 {"speaker": "female", "speed": 1.0, ...}
        Returns:
            bool: TTS指令是否成功发送
        """
        pass

    @abstractmethod
    async def start_asr_monitoring(self):
        """
        启动ASR监听
        持续监听音频输入，通过asr_callback推送识别结果
        """
        pass

    @abstractmethod
    async def stop_asr_for_call(self, call_id: str) -> bool:
        """
        停止指定通话的ASR监听
        Args:
            call_id: 通话ID
        Returns:
            bool: 是否成功停止
        """
        pass

    async def run(self):
        """启动ASR驱动"""
        await self.start_asr_monitoring()

# 模拟实现（用于测试）
class MockDeviceDriver(DeviceDriver):
    def __init__(self):
        super().__init__()
        self._monitoring = False
        self.main_loop = None  # 主线程事件循环
    

    async def dial(self, phone_number: str, call_id: str = "") -> bool:
        print(f"[MockDeviceDriver] 拨号: {phone_number}, call_id: {call_id}")
        if not call_id:
            call_id = f"call_{len(self.call_states)+1:03d}"
        self.call_states[call_id] = {"phone_number": phone_number, "status": "dialing"}
        return True

    async def hangup(self, call_id: str) -> bool:
        print(f"[MockDeviceDriver] 挂断: {call_id}")
        if call_id in self.call_states:
            self.call_states[call_id]["status"] = "hangup"
        return True

    async def get_call_status(self, call_id: str) -> Dict[str, Any]:
        return self.call_states.get(call_id, {"status": "unknown"})

    async def start_hardware_monitoring(self):
        self._monitoring = True
        while self._monitoring:
            await asyncio.sleep(1)
            # 模拟硬件事件
            for call_id, state in self.call_states.items():
                if state.get("status") == "dialing":
                    state["status"] = "connected"
                    event = {
                        "version": self.version,
                        "type": "event",
                        "call_id": call_id,
                        "event_type": "call_connected",
                        "detail": {"phone_number": state.get("phone_number")},
                        "timestamp": int(asyncio.get_event_loop().time() * 1000)
                    }
                    if self.event_callback:
                        await self.event_callback(event)
                    await asyncio.sleep(50)  # 模拟通话时间
                    state["status"] = "ended"
                    event = {
                        "version": self.version,
                        "type": "event",
                        "call_id": call_id,
                        "event_type": "call_ended",
                        "detail": {},
                        "timestamp": int(asyncio.get_event_loop().time() * 1000)
                    }
                    if self.event_callback:
                        await self.event_callback(event)

class MockASRDriver(ASRDriver):
    def __init__(self):
        super().__init__()
        self._monitoring = False

    async def send_tts(self, call_id: str, text_data: List[Dict[str, Any]],
                       voice_params: Optional[Dict[str, Any]] = None) -> bool:
        print(f"[MockASRDriver] TTS发送: call_id={call_id}, text_data={text_data}, voice_params={voice_params}")

        try:
            # ✅ 让同步的 generate_tts_wav_and_play 在后台线程运行
            await asyncio.to_thread(
                generate_tts_wav_and_play,
                text_data,
                '10.16.80.147',
                50000
            )
            return True
        except Exception as e:
            print(f"[MockASRDriver] TTS播放失败: {e}")
            return False
    
    async def start_asr_monitoring(self):
        """
        实时麦克风采集并推送ASR识别结果（调用 asrexample.py 里的逻辑）
        """
        import pyaudio
        import time
        import threading
        import queue
        from asr import speech_recognizer
        from common import credential

        APPID = "1370866869"
        SECRET_ID = 
        SECRET_KEY = 
        ENGINE_MODEL_TYPE = "16k_zh"
        SLICE_SIZE = 6400

        class Listener(speech_recognizer.SpeechRecognitionListener):
            def __init__(self, call_id, outer):
                self.call_id = call_id
                self.outer = outer
            def on_sentence_end(self, response):
                import asyncio
                text = response.get("result", "")
                # 兼容腾讯云详细返回结构，自动提取 voice_text_str
                if isinstance(text, dict):
                    text_str = text.get("voice_text_str", "")
                else:
                    text_str = text
                if text_str and self.outer.asr_callback:
                    asr_result = {
                        "version": self.outer.version,
                        "type": "receive_text",
                        "call_id": self.call_id,
                        "from": "customer",
                        "text_data": [
                            {"text": text_str, "emo": ""}
                        ],
                        "timestamp": int(time.time() * 1000)
                    }
                    coro = self.outer.asr_callback(asr_result)
                    loop = getattr(self.outer, 'main_loop', None)
                    if loop and loop.is_running():
                        loop.call_soon_threadsafe(asyncio.create_task, coro)
                    else:
                        print("[ASR回调] 主线程无事件循环，ASR结果未推送")

        def asr_thread(call_id, outer):
            listener = Listener(call_id, outer)
            cred = credential.Credential(SECRET_ID, SECRET_KEY)
            recognizer = speech_recognizer.SpeechRecognizer(
                APPID, cred, ENGINE_MODEL_TYPE, listener)
            recognizer.set_filter_modal(1)
            recognizer.set_filter_punc(1)
            recognizer.set_filter_dirty(1)
            recognizer.set_need_vad(1)
            recognizer.set_voice_format(1)
            recognizer.set_word_info(1)
            recognizer.set_convert_num_mode(1)

            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            CHUNK = SLICE_SIZE

            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            recognizer.start()
            try:
                while self._monitoring:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    recognizer.write(data)
                    time.sleep(0.02)
            finally:
                recognizer.stop()
                stream.stop_stream()
                stream.close()
                p.terminate()

        self._monitoring = True
        # 这里只模拟一个call_id，实际可根据active_calls动态管理
        if not self.active_calls:
            self.active_calls["call_001"] = {"status": "active"}
        threads = []
        for call_id in self.active_calls.keys():
            t = threading.Thread(target=asr_thread, args=(call_id, self), daemon=True)
            t.start()
            threads.append(t)
        # 保持主协程活跃
        while self._monitoring:
            await asyncio.sleep(1)

    async def stop_asr_for_call(self, call_id: str) -> bool:
        print(f"[MockASRDriver] 停止ASR: {call_id}")
        self.active_calls.pop(call_id, None)
        return True

class PhoneModuleServer:
    def __init__(self, host="0.0.0.0", port=8765, version="1.1"):
        self.host = host
        self.port = port
        self.version = version
        self.ws_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.call_states: Dict[str, Dict[str, Any]] = {}
        # 使用模拟驱动，后续可替换为真实驱动
        self.device_driver = MockDeviceDriver()
        self.device_driver.event_callback = self.on_hardware_event
        self.asr_driver = MockASRDriver()
        self.asr_driver.asr_callback = self.on_asr_result

    async def ws_handler(self, websocket, path):
        self.ws_clients.add(websocket)
        try:
            async for message in websocket:
                await self.handle_platform_msg(json.loads(message), websocket)
        finally:
            self.ws_clients.remove(websocket)

    async def handle_platform_msg(self, data: Dict[str, Any], websocket):
        msg_type = data.get("type")
        call_id = data.get("call_id", "")
        if msg_type == "control":
            action = data.get("action")
            params = data.get("params", {})
            print(f"[平台指令] control: {action}, call_id={call_id}, params={params}")
            
            if action == "dial":
                phone_number = params.get("phone_number", "")
                success = await self.device_driver.dial(phone_number, call_id)
                if success:
                    print(f"[平台指令] 拨号指令已发送给硬件驱动")
            elif action == "hangup":
                success = await self.device_driver.hangup(call_id)
                if success:
                    print(f"[平台指令] 挂断指令已发送给硬件驱动")
                    await self.asr_driver.stop_asr_for_call(call_id)
            elif action == "status_query":
                status = await self.device_driver.get_call_status(call_id)
                print(f"[平台指令] 查询状态: {status}")
                
        elif msg_type == "send_text":
            text_data = data.get("text_data", [])
            voice_params = data.get("voice_params", {})
            print(f"[平台指令] send_text: call_id={call_id}, text_data={text_data}, voice_params={voice_params}")
            
            success = await self.asr_driver.send_tts(call_id, text_data, voice_params)
            if success:
                print(f"[平台指令] TTS指令已发送给语音驱动")

    async def on_hardware_event(self, event):
        await self.broadcast_to_platform(event)

    async def on_asr_result(self, asr_result):
        await self.broadcast_to_platform(asr_result)

    async def broadcast_to_platform(self, event):
        for ws in list(self.ws_clients):
            try:
                await ws.send(json.dumps(event,ensure_ascii=False))
            except Exception as e:
                print(f"[广播异常] {e}")

    def run(self):
        async def main():
            server = await websockets.serve(self.ws_handler, self.host, self.port)
            print(f"PhoneModuleServer running at ws://{self.host}:{self.port}")
            self.main_loop = asyncio.get_running_loop()
            self.asr_driver.main_loop = self.main_loop
            await asyncio.gather(
                self.device_driver.run(),
                self.asr_driver.run(),
                server.wait_closed(),
            )
        asyncio.run(main())

if __name__ == "__main__":
    server = PhoneModuleServer(args.host, args.port, args.version)
    server.run()
