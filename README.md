# Auto Annotation Tool

Reolinkカメラ動体検知 × SAM3自動セグメンテーションツール

## 概要

Reolinkカメラからの映像を監視し、動体を検知すると自動的にスナップショットを撮影して、SAM3（Segment Anything Model 3）を使用してオブジェクトのセグメンテーションとアノテーションを自動実行するツールです。ハムスターなどの小動物の行動記録と学習データ収集に最適です。

## 主な機能

- **自動動体検知**: カメラAPI方式またはソフトウェア映像差分方式
- **自動撮影**: 動体検知時にスナップショット自動取得
- **SAM3セグメンテーション**: テキストプロンプトベースのオブジェクト検出・セグメンテーション
- **データ自動保存**: 画像、セグメンテーションマスク、JSONアノテーションを自動保存
- **複数実行モード**: 自動連続モード、1回実行モード、各種テストモード

## 技術スタック

- **カメラ通信**: Reolink HTTP/RTSP API
- **動体検知**: カメラ内蔵API / OpenCV映像差分
- **セグメンテーション**: SAM3 (Segment Anything Model 3)
- **画像処理**: OpenCV, Pillow
- **設定管理**: YAML

## ディレクトリ構成

```
auto_annotation/
├── main.py                  # メインプログラム
├── config.yaml              # 設定ファイル
├── api_server.py            # APIサーバー（オプション）
├── pyproject.toml           # プロジェクト依存関係 (uv)
├── src/
│   ├── camera.py            # Reolinkカメラ通信
│   ├── motion_detector.py   # 動体検知
│   ├── annotator.py         # SAM3アノテーション
│   └── utils.py             # ユーティリティ関数
├── sam3/                    # SAM3モデルサブモジュール
└── output/                  # 出力ディレクトリ（自動生成）
    ├── images/              # 撮影画像
    ├── masks/               # セグメンテーションマスク
    └── annotations/         # JSONアノテーション
```

## セットアップ

### 必要要件

- Python 3.8+
- CUDA対応GPU（推奨、CPUでも動作可能）
- Reolinkカメラ（RLC-510Aなど）
- 十分なストレージ容量

### インストール

1. 依存パッケージのインストール:
```bash
cd auto_annotation
pip install torch torchvision opencv-python requests pyyaml pillow numpy
```

2. SAM3モデルのセットアップ:
```bash
cd sam3
pip install -e .
cd ..
```

### 設定

`config.yaml` を編集して、カメラとアノテーションの設定を行います。

#### カメラ設定

```yaml
camera:
  ip: "192.168.31.85"
  http_port: 8000
  username: "admin"
  password: "your_password"  # 変更してください
  rtsp_port: 554
  main_stream_suffix: "/h264Preview_01_main"
  sub_stream_suffix: "/h264Preview_01_sub"
```

#### 動体検知設定

```yaml
motion_detection:
  enabled: true
  method: "api"               # "api" (カメラ内蔵) or "software" (映像差分)
  polling_interval: 0.5       # 秒
  cooldown: 5.0               # 検知後の待機時間（秒）
  threshold: 20               # ソフトウェア検知の差分閾値
```

#### SAM3設定

```yaml
sam3:
  model_type: "vit_b"
  checkpoint_path: "./checkpoints/sam3_vit_b.pth"
  device: "cuda"              # "cuda" or "cpu"
  prompt: "hamster"           # 検出対象のテキストプロンプト
  confidence_threshold: 0.5
```

#### 出力設定

```yaml
output:
  base_dir: "./output"
  save_images: true
  save_masks: true
  save_json: true
  filename_format: "%Y%m%d_%H%M%S"
  max_captures: 1000          # 最大取得枚数（0で無制限）
```

## 使い方

### 基本的な実行

自動モードで起動（動体検知→撮影→アノテーションを自動実行）:

```bash
python main.py
```

### 実行モード

#### 自動モード（デフォルト）
```bash
python main.py --mode auto
```

#### 1回だけ実行
```bash
python main.py --mode auto --once
```

#### カメラ接続テスト
```bash
python main.py --mode test-camera
```

#### SAM3モデルテスト
```bash
python main.py --mode test-sam
```

#### 動体検知テスト
```bash
python main.py --mode test-motion
```

### オプション

- `--config, -c`: 設定ファイルのパス（デフォルト: `config.yaml`）
- `--mode, -m`: 実行モード
- `--once`: 自動モードで1回だけ実行して終了
- `--verbose, -v`: 詳細ログを表示

### 例

```bash
# カスタム設定ファイルを使用
python main.py --config my_config.yaml

# 詳細ログを表示
python main.py --verbose

# カメラテストのみ実行
python main.py --mode test-camera
```

## 出力形式

### 画像
- フォーマット: JPG
- パス: `output/images/{timestamp}.jpg`

### マスク
- フォーマット: PNG（グレースケール、0/255）
- パス: `output/masks/{timestamp}_mask_{index}.png`

### アノテーション (JSON)

```json
{
  "image_id": "20250101_120000",
  "timestamp": "2025-01-01T12:00:00.123456",
  "image_path": "images/20250101_120000.jpg",
  "prompt": "hamster",
  "objects": [
    {
      "id": 0,
      "label": "hamster",
      "confidence": 0.87,
      "bbox": [x, y, width, height],
      "mask_path": "masks/20250101_120000_mask_0.png"
    }
  ]
}
```

## ワークフロー

1. カメラ映像を監視
2. 動体検知（カメラAPIまたは映像差分）
3. スナップショット撮影
4. SAM3でセグメンテーション
5. 画像・マスク・JSONを保存
6. クールダウン期間待機
7. ステップ1に戻る

## トラブルシューティング

### カメラに接続できない

- カメラのIPアドレス、ポート、認証情報を確認
- ネットワーク接続を確認（同一ネットワーク上にあるか）
- `--mode test-camera` でテスト実行

### SAM3モデルがロードできない

- `sam3/` ディレクトリが正しくセットアップされているか確認
- チェックポイントファイルのパスが正しいか確認
- GPU メモリが不足している場合は `device: "cpu"` に変更
- `--mode test-sam` でテスト実行

### 動体検知が動作しない

- `config.yaml` の `method` 設定を確認
- `--mode test-motion` でテスト実行
- API方式で失敗する場合は、software方式を試す
- カメラの動体検知設定がONになっているか確認

### セグメンテーション精度が低い

- `prompt` を具体的な名称に変更（例: "hamster" → "golden hamster"）
- `confidence_threshold` を調整
- 照明条件を改善

## 連携ツール

このツールの出力は `annotation_checker` ツールで品質チェックできます:

```bash
cd ../annotation_checker
uv run checker.py
```

## パフォーマンス

- **GPU使用時**: 約1-2秒/画像
- **CPU使用時**: 約5-10秒/画像
- **推奨メモリ**: 8GB以上（GPU）、16GB以上（CPU）

## ライセンス

このプロジェクトは個人使用を想定しています。

## 注意事項

- カメラのパスワードは `config.yaml` に平文で保存されます。ファイルの取り扱いに注意してください
- 長時間の実行時は `max_captures` の設定を推奨します
- CUDA対応GPUがない場合、`device: "cpu"` に設定してください（処理速度は低下します）
- ストレージ容量に注意してください（1000枚で約500MB-2GB）
