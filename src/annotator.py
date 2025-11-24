import torch
import numpy as np
from PIL import Image
import logging
import time
import cv2

logger = logging.getLogger(__name__)

class SAM3Annotator:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.model = None
        self.processor = None
        self.prompt = config.get('prompt', 'person')
        
    def load_model(self):
        """
        SAM3モデルをロードする
        """
        logger.info("Loading SAM3 model...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            
            logger.info("SAM3 model loaded successfully.")
            return True
        except ImportError as e:
            logger.error(f"Failed to import SAM3: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Error loading SAM3 model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def annotate(self, image_bgr, prompt=None):
        """
        画像に対してアノテーションを実行する
        Args:
            image_bgr: OpenCV形式 (BGR) の画像
            prompt: (Optional) テキストプロンプト。指定がない場合は初期化時のものを使用。
        Returns:
            dict: {
                "masks": list of numpy arrays,
                "boxes": list of [x, y, w, h],
                "scores": list of float,
                "labels": list of str
            }
        """
        if self.model is None:
            logger.error("Model not loaded.")
            return None

        try:
            # OpenCV (BGR) -> PIL (RGB)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # 使用するプロンプトを決定
            target_prompt = prompt if prompt else self.prompt
            logger.info(f"Annotating with prompt: {target_prompt}")

            # 推論実行
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=target_prompt
            )
            
            # 結果の整形
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Tensor -> Numpy conversion helper
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                if isinstance(x, list):
                    # If list contains tensors, convert them
                    if len(x) > 0 and isinstance(x[0], torch.Tensor):
                        return np.array([t.detach().cpu().numpy() for t in x])
                    return np.array(x)
                return np.array(x)

            masks = to_numpy(masks)
            boxes = to_numpy(boxes)
            scores = to_numpy(scores)
            
            # スコアフィルタリング
            threshold = self.config.get('confidence_threshold', 0.5)
            
            filtered_results = {
                "masks": [],
                "boxes": [],
                "scores": [],
                "labels": []
            }
            
            if scores is not None:
                scores_np = scores # Already numpy
                for i, score in enumerate(scores_np):
                    if score >= threshold:
                        mask = masks[i]
                        bbox = boxes[i] # [x1, y1, x2, y2]
                        
                        # オプション: マスクからbboxを再計算
                        if self.config.get('recalculate_bbox_from_mask', False):
                            # マスクからbboxを計算
                            # maskは (H, W) のboolまたはuint8
                            y_indices, x_indices = np.where(mask > 0)
                            if len(x_indices) > 0 and len(y_indices) > 0:
                                x1 = np.min(x_indices)
                                y1 = np.min(y_indices)
                                x2 = np.max(x_indices)
                                y2 = np.max(y_indices)
                                bbox = np.array([x1, y1, x2, y2])
                            else:
                                # マスクが空の場合は元のbboxを使うか、無効とする
                                # ここでは元のbboxを使うことにする
                                pass

                        # bbox形式変換: [x1, y1, x2, y2] -> [x, y, w, h]
                        # x, y は左上座標
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Pythonのfloat型に変換してリスト化
                        bbox_xywh = [float(x1), float(y1), float(w), float(h)]
                        
                        filtered_results["masks"].append(mask)
                        filtered_results["boxes"].append(bbox_xywh)
                        filtered_results["scores"].append(float(score))
                        filtered_results["labels"].append(target_prompt)
            
            return filtered_results

        except Exception as e:
            logger.error(f"Annotation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
