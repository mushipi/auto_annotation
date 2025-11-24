import time
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, config, camera):
        self.config = config
        self.camera = camera
        self.method = config.get('method', 'api')
        self.threshold = config.get('threshold', 20)
        self.last_frame = None
        
        logger.info(f"Motion Detector initialized. Method: {self.method}")

    def check_motion(self):
        """
        動体検知を確認する
        Returns: True if motion detected, False otherwise
        """
        # APIメソッドが指定されていても、カメラ側でAPIが使えない場合はソフトウェア検知にフォールバック
        if self.method == 'api':
            if self.camera.api_available:
                return self._check_api()
            else:
                # APIが使えないのでソフトウェア検知を使用
                # 初回のみログを出すなどの制御があってもいいが、ここではシンプルに実行
                return self._check_software()
        elif self.method == 'software':
            return self._check_software()
        else:
            logger.warning(f"Unknown motion detection method: {self.method}")
            return False

    def _check_api(self):
        """
        カメラAPIを使用した動体検知
        """
        # APIが失敗した場合はソフトウェア検知にフォールバックするロジックも検討可能だが、
        # ここではシンプルにAPIの結果を返す
        return self.camera.get_motion_state()

    def _check_software(self):
        """
        映像差分による簡易動体検知
        """
        # サブストリームからフレーム取得（軽量化のため）
        frame = self.camera.get_rtsp_frame(main_stream=False)
        
        if frame is None:
            return False

        # グレースケール変換とブラー
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.last_frame is None:
            self.last_frame = gray
            return False

        # 差分計算
        frame_delta = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # 差分領域の面積を計算
        motion_detected = False
        if np.sum(thresh) > 10000: # 閾値は調整が必要
             motion_detected = True

        self.last_frame = gray
        return motion_detected
