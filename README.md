# anime-frame-qa

アニメフレームの品質管理パイプライン。

アニメ画像・動画のアーティファクト（フリッカー、ノイズ、輪郭途切れ、フレーム間色ずれ）を検出・補正します。

## 特徴

### OpenCVの利用

| モジュール       | 内容                                                                       | 画像  | 動画  | オプション                                                  |
| ----------- | ------------------------------------------------------------------------ | :-: | :-: | ------------------------------------------------------ |
| **フリッカー抑制** | ヒストグラム比較による検出 + EMA時間方向平滑化                                               |  -  |  o  | `--deflicker`                                          |
| **ノイズ抑制**   | バイラテラルフィルタ（Gaussianノイズ）/ Non-Local Means（モスキートノイズ）/ バンディング除去（グラデーションの段差） |  o  |  o  | `--denoise [--denoise-method bilateral\|nlm\|banding]` |
| **輪郭抽出**    | Canny + Zhang-Suen細線化 + 途切れ検出・可視化                                        |  o  |  o  | `--extract-edges`                                      |
| **色一貫性**    | 直近Nフレームのヒストグラムを参照にCDF-LUTで補正                                             |  -  |  o  | `--color-consistency`                                  |

### 既存CNNモデル活用（オプショナル）

| モジュール    | モデル                             | 用途             | 画像  | 動画  | オプション                               |
| -------- | ------------------------------- | -------------- | :-: | :-: | ----------------------------------- |
| **背景除去** | isnet-anime (rembg経由)           | 前景/背景分離        |  o  |  -  | `--remove-bg`                       |
| **修復**   | LaMa (simple-lama-inpainting経由) | 問題領域のインペインティング |  o  |  -  | `--inpaint --inpaint-mask mask.png` |

全CNNモデルはCPU実行可能。`uv sync --extra cnn` でインストール。

> **注意**: `rembg` が `opencv-python` を引き込み、`opencv-contrib-python-headless` と競合する場合があります。Zhang-Suen細線化（`--extract-edges`）が動かなくなった場合は以下を実行してください：
> ```bash
> uv pip install --force-reinstall opencv-contrib-python-headless
> ```

## インストール

```bash
# コア（OpenCVのみ、軽量）
uv sync

# CNNモデル付き
uv sync --extra cnn

# W&B実験管理付き
uv sync --extra wandb
```

## 使い方

### 画像の処理

```bash
# Gaussianノイズ除去
uv run pipeline process input.png --denoise --denoise-method bilateral -o output.png

# バンディング除去
uv run pipeline process input.png --denoise --denoise-method banding -o output.png

# モスキートノイズ除去
uv run pipeline process input.png --denoise --denoise-method nlm -o output.png

# 輪郭抽出・途切れ検出（細線化済み画像 + ギャップ位置を赤丸でマーク）
uv run pipeline process input.png --extract-edges -o output.png
# → output.png にギャップ可視化済みのthinned画像を出力
# → CLI に "edges: N gap(s) detected" を表示

# 全コアモジュール適用
uv run pipeline process input.png --all -o output.png
```

### 画像のバッチ処理

ディレクトリを渡すと内部の画像ファイルを一括処理します。各画像は独立して処理され、動画として扱われません。

```bash
# ディレクトリ内の全画像にノイズ抑制を適用
uv run pipeline process frames/ --denoise -o output/
```

対応フォーマット: PNG / JPEG / BMP / TIFF / WebP

### 動画の処理

> **出力コーデック**: `mp4v` 固定のため、VLC等のメディアプレイヤーで再生してください。Webブラウザでの再生は非対応です。

```bash
# フリッカー抑制 + ノイズ抑制
uv run pipeline process input.mp4 --deflicker --denoise -o output.mp4

# フレーム間色一貫性
uv run pipeline process input.mp4 --color-consistency -o output.mp4

# 全コアモジュール適用
uv run pipeline process input.mp4 --all -o output.mp4
```

### YAML設定ファイルの利用

```bash
uv run pipeline process input.mp4 --config config/my_settings.yaml -o output.mp4
```

> **注意**: `--config` を指定すると、他のCLIフラグ（`--denoise` など）は**すべて無視**されます。YAMLファイルに全設定を記述してください。

```yaml
# 設定例
deflicker: true
denoise: true
denoise_method: bilateral   # bilateral | nlm | banding
extract_edges: false
color_consistency: true
```

> **サブパラメータについて**: `d`・`sigma_color` などの詳細パラメータはYAMLからは設定できません。これらを変更したい場合は `sweep` コマンドを使用してください。

### パラメータスイープ（W&B連携オプション）

**1枚の画像に対して複数のノイズ抑制パラメータセットを試行し、PSNR/SSIMで効果を比較します。**
バッチ処理（ディレクトリ入力）とは異なり、入力は常に画像1枚です。`denoise` 専用で、`deflicker` などフレーム間処理はスイープ対象外です。

```bash
# ローカルスイープ
uv run pipeline sweep input.png config/sweep_example.yaml -o sweep_results/

# W&Bログ付き
uv run pipeline sweep input.png config/sweep_example.yaml -o sweep_results/ --wandb-project anime-qa
```

指定可能なパラメータとデフォルト値:

| パラメータ             | デフォルト       | 対象メソッド    |
| ----------------- | ----------- | --------- |
| `denoise_method`  | `bilateral` | 共通        |
| `d`               | `9`         | bilateral |
| `sigma_color`     | `75.0`      | bilateral |
| `sigma_space`     | `75.0`      | bilateral |
| `h`               | `10.0`      | nlm       |
| `template_window` | `7`         | nlm       |
| `search_window`   | `21`        | nlm       |
| `blur_radius`     | `5`         | banding   |
| `edge_low`        | `30`        | banding   |
| `edge_high`       | `90`        | banding   |

### 背景除去・インペインティング（CNN extras 必要）

```bash
# 背景除去（isnet-anime、初回実行時にモデル自動ダウンロード ~170MB）
uv run pipeline process input.png --remove-bg -o output.png

# インペインティング（LaMa）— マスク画像で修復対象領域を指定（白=修復）
uv run pipeline process input.png --inpaint --inpaint-mask mask.png -o output.png
```

## アーキテクチャ

```
src/anime_frame_qa/
  cli.py              CLI（click）
  pipeline.py         モジュール統合・実行制御
  io.py               画像/動画 I/O（チャンク読み込み対応）
  config.py           YAML設定ファイル読み込み
  metrics.py          品質メトリクス（PSNR / SSIM）
  sweep.py            パラメータスイープ + W&B連携
  modules/
    deflicker.py      フリッカー検出・抑制
    denoise.py        ノイズ抑制（bilateral / NLM）
    edge.py           輪郭抽出・細線化・途切れ検出
    color.py          フレーム間色一貫性
    background.py     背景除去（isnet-anime、オプショナル）
    inpaint.py        インペインティング（LaMa、オプショナル）
```

## アルゴリズム詳細

### フリッカー抑制
グレースケールヒストグラムのBhattacharyya距離でフレーム間の明るさ変動を検出。閾値を超えた場合、直近フレームの指数移動平均(EMA)に基づくチャネル別補正を適用。輝度変動が小さい（±10%以下）場合は検出されないことがあります。

### ノイズ抑制
- **バイラテラルフィルタ** (`bilateral`): エッジを保持しつつランダムなGaussianノイズを除去。バンディングには不向き（ステップ境界をエッジとして保護するため）。
- **Non-Local Means** (`nlm`): エッジ周辺の高周波アーティファクト（モスキートノイズ）に有効。JPEG圧縮由来のリンギングなど。
- **バンディング除去** (`banding`): Cannyエッジ検出で線画・輪郭を保護し、平坦・グラデーション領域のみGaussianブラーで平滑化。空・肌などのステップ状段差に有効。

### 輪郭抽出・途切れ検出
1. ガウシアンブラー後のCanny エッジ検出
2. Zhang-Suen形態学的細線化（1ピクセル幅）
3. 近傍カウントによる端点検出
4. 距離範囲内の端点ペアをギャップとして検出
5. 検出したギャップ端点を赤丸でマークして出力（診断ツールとして位置づけ）

### 色一貫性
直近Nフレームのスライディングウィンドウから参照を構築し、CDF-LUTベースのヒストグラムマッチングを適用。

## テスト

```bash
uv run pytest tests/ -v
```

## 技術スタック

- **Python 3.12+** / **uv** パッケージマネージャ
- **OpenCV** (opencv-contrib-python-headless) — コア画像処理
- **NumPy** — 配列演算
- **click** — CLI
- **PyYAML** — 設定ファイル
- **W&B** (オプショナル) — 実験管理・パラメータ最適化
- **rembg / simple-lama-inpainting** (オプショナル) — CNNベース処理
