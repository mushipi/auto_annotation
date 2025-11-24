import cv2
import requests
import time
import logging
import urllib3
import os

# SSL警告を抑制
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# WSL/Linux環境でのRTSP接続安定化のためTCPを強制
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

logger = logging.getLogger(__name__)

class ReolinkCamera:
    def __init__(self, config):
        self.config = config
        self.ip = config['ip']
        self.http_port = config['http_port']
        self.username = config['username']
        self.password = config['password']
        
        self.protocol = config.get('protocol', 'http')
        self.base_url = f"{self.protocol}://{self.ip}:{self.http_port}/cgi-bin/api.cgi"
        
        # RTSP URLの構築
        # hamcam4に合わせてquoteを使用しない
        safe_password = self.password
        self.main_stream_url = f"rtsp://{self.username}:{safe_password}@{self.ip}:{config['rtsp_port']}{config['main_stream_suffix']}"
        self.sub_stream_url = f"rtsp://{self.username}:{safe_password}@{self.ip}:{config['rtsp_port']}{config['sub_stream_suffix']}"
        
        self.api_available = False

    def connect(self):
        """
        カメラへの接続テスト
        1. API (HTTPS/HTTP, 各ポート) を試す
        2. 失敗したら RTSP を試す
        """
        # 1. API接続試行
        if self._connect_api():
            self.api_available = True
            return True
            
        logger.warning("API connection failed. Trying RTSP...")
        
        # 2. RTSP接続試行
        if self._connect_rtsp():
            logger.info("RTSP connection successful. Falling back to RTSP-only mode.")
            self.api_available = False
            return True
            
        logger.error("All connection attempts (API & RTSP) failed.")
        return False

    def _connect_api(self):
        """API接続を総当たりで試す"""
        candidates = []
        candidates.append(('https', self.http_port))
        candidates.append(('http', self.http_port))
        if self.http_port != 443: candidates.append(('https', 443))
        if self.http_port != 80: candidates.append(('http', 80))

        for protocol, port in candidates:
            logger.info(f"Trying API: {protocol}://{self.ip}:{port} ...")
            if self._try_api_connection(protocol, port):
                self.protocol = protocol
                self.http_port = port
                self.base_url = f"{protocol}://{self.ip}:{port}/cgi-bin/api.cgi"
                logger.info(f"API Connected! Using {protocol}://{self.ip}:{port}")
                return True
        return False

    def _connect_rtsp(self):
        """RTSP接続確認"""
        try:
            # URLをログ出力（パスワードは隠す）
            masked_url = self.main_stream_url.replace(self.password, "****")
            logger.info(f"Connecting to RTSP: {masked_url}")
            
            cap = cv2.VideoCapture(self.main_stream_url)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return True
            else:
                logger.warning("VideoCapture not opened")
        except Exception as e:
            logger.error(f"RTSP connection error: {e}")
        return False

    def _try_api_connection(self, protocol, port):
        try:
            url = f"{protocol}://{self.ip}:{port}/cgi-bin/api.cgi"
            payload = [{"cmd": "GetDevInfo", "action": 0, "param": {}}]
            response = requests.post(
                url, json=payload, auth=(self.username, self.password),
                timeout=3, verify=False
            )
            return response.status_code == 200
        except:
            return False

    def get_snapshot(self):
        """
        スナップショットを取得 (API優先、失敗時はRTSP)
        """
        if self.api_available:
            try:
                url = f"{self.base_url}?cmd=Snap&channel=0&user={self.username}&password={self.password}"
                response = requests.get(url, timeout=10, verify=False)
                if response.status_code == 200:
                    import numpy as np
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"API Snapshot failed: {e}")
        
        # Fallback to RTSP
        logger.info("Using RTSP for snapshot...")
        return self.get_rtsp_frame(main_stream=True)

    def get_motion_state(self):
        """
        動体検知状態を取得 (APIのみ)
        """
        if not self.api_available:
            return False

        try:
            payload = [{"cmd": "GetMdState", "action": 0, "param": {"channel": 0}}]
            response = requests.post(
                self.base_url, json=payload, auth=(self.username, self.password),
                timeout=2, verify=False
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('value', {}).get('state', 0) == 1
            return False
        except:
            return False

    def get_rtsp_frame(self, main_stream=False):
        """
        RTSPストリームから1フレーム取得
        """
        url = self.main_stream_url if main_stream else self.sub_stream_url
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
