# ESM-2 ファインチューニング & DMS スクリーニングパイプライン

タンパク質の機能活性を予測するため、Deep Mutational Scanning（DMS）データ上で **ESM-2** タンパク質言語モデルをファインチューニングする、スケーラブルで再現性の高い Nextflow パイプラインです。大規模な変異ライブラリを複数のデータセットにまたがって並列に処理するハイスループットスクリーニングを目的として設計されています。

---

## 概要

本パイプラインは、DMS の点変異データ上で ESM-2（Evolutionary Scale Modeling 2）をファインチューニングし、配列から機能への関係性を学習します。学習・ハイパーパラメータ最適化・推論の一連のワークフローを **Nextflow** でラップし、**Docker** コンテナ化することで、数百に及ぶタンパク質データセットを一貫性・再現性を保ちながら大規模並列実行できます。

### 主な特長

- **ESM-2 ファインチューニング** — Facebook の ESM-2 トランスフォーマーモデル（例: `esm2_t36_3B_UR50D`）に、平均プーリング + MLP 回帰ヘッドを搭載し、DMS スコア予測を行います。
- **LoRA パラメータ効率的ファインチューニング** — PEFT による Low-Rank Adaptation（LoRA）をサポート。学習パラメータを大幅に削減しつつ、モデル品質を維持します。Attention の `query`、`key`、`value`、`dense` モジュールを対象とします。
- **自動ハイパーパラメータ最適化（HPO）** — **Optuna** を統合し、学習率・Weight Decay・LoRA ランク・ドロップアウトを自動探索。有望でないトライアルはプルーニング（枝刈り）されます。
- **3 ステージワークフロー** — (1) 短いエポックでの HPO 探索、(2) 最適パラメータによる最終フル学習、(3) テストデータでのバッチ推論。
- **WandB リアルタイムトラッキング** — すべての学習・HPO トライアル・推論ジョブが **Weights & Biases** に記録されます。メトリクス、アーティファクト、モデルチェックポイント、予測分布などを含みます。
- **Docker コンテナ化** — CUDA 12.8 + Python 3.13 + PyTorch 2.6 の Docker イメージにより、環境の安定性とマシン間の再現性を保証します。
- **マルチデータセット並列実行** — メタデータ CSV を介して数十の DMS データセットを同時処理。GPU 並列度は設定可能です。
- **混合精度（AMP）** — BF16/FP16 の自動混合精度学習と勾配スケーリングにより、メモリ効率と速度を向上させます。
- **アーリーストッピング** — 検証データのスピアマン相関が頭打ちになると学習を停止し、過学習を防ぎ計算資源を節約します。
- **詳細なレポート** — Nextflow の実行レポート、タイムライン、DAG フロー図を自動生成します。

---

## パイプラインアーキテクチャ

```
samples_metadata.csv
        │
        ▼
┌───────────────────┐
│  SAMPLE_PARSING   │  メタデータ CSV を解析 → (id, データセットCSV) のチャネルへ
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   HPO_SEARCH      │  Optuna によるハイパーパラメータ探索（短いエポック）
│  （スキップ可）    │  出力: best_hpo_params.json
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   TRAIN_FINAL     │  最適 HPO パラメータでのフル学習
│                   │  出力: モデル重み、scaler、テスト分割、wandb_run_id
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│     PREDICT       │  テストセットでの推論 → predictions CSV
└───────────────────┘
```

---

## プロジェクト構成

```
.
├── main.nf                  # Nextflow エントリーポイント & ワークフロー統合
├── nextflow.config          # パイプライン設定（パラメータ、プロファイル、リソース）
├── Dockerfile               # CUDA 12.8 + Python 3.13 + PyTorch 2.6 コンテナ
├── requirements.txt         # Python 依存パッケージ
├── samples_metadata.csv     # 入力データセットマニフェスト（id, sample, info, data パス）
├── modules/
│   ├── hpo_search.nf        # Optuna HPO 探索プロセス
│   ├── train_final.nf       # 最終学習プロセス（HPO パラメータを読み込み）
│   ├── train.nf             # スタンドアロン学習プロセス
│   └── predict.nf           # 推論 / 予測プロセス
└── src/
    ├── train.py             # 学習スクリプト（HPO + 最終学習）
    ├── predict.py           # 推論スクリプト
    ├── model.py             # 平均プーリング回帰ヘッド付き ESM-2 モデル
    └── dataset.py           # データセットクラス & データ読み込みユーティリティ
```

---

## 前提条件

- **Nextflow** ≥ 23.04（DSL 2）
- **Docker** と NVIDIA Container Toolkit（GPU サポート）
- **NVIDIA GPU**（CUDA ≥ 12.8 互換ドライバ）
- **W&B アカウント** と実験管理用 API キー

---

## インストール

### 1. Docker イメージのビルド

```bash
docker build -t my-gpu-app:v1.2 .
```

イメージは `nvidia/cuda:12.8.0-runtime-ubuntu22.04` をベースに、Python 3.13、PyTorch 2.6、およびすべての依存パッケージが `uv` でインストールされています。

### 2. WandB の設定

```bash
export WANDB_API_KEY="your_wandb_api_key"
```

Nextflow がこれをシークレットとして各プロセスに注入します。

### 3. データの準備

以下のカラムを持つ `samples_metadata.csv` を作成します:

| カラム | 説明 |
|--------|------|
| `id` | データセットの一意の識別子（出力ディレクトリ名に使用） |
| `sample` | サンプル / タンパク質名 |
| `info` | 追加の説明（研究参考文献など） |
| `data` | DMS データ CSV の絶対パス |

各 DMS データ CSV には、少なくとも以下のカラムが必要です:
- `mutated_sequence` — タンパク質のアミノ酸配列
- `DMS_score` — 機能活性スコア（回帰の目的変数）

---

## 使い方

### 基本的な実行

```bash
nextflow run main.nf \
    --input samples_metadata.csv \
    --output_dir ./output \
    -profile docker
```

### HPO をスキップ（直接学習）

```bash
nextflow run main.nf \
    --input samples_metadata.csv \
    --output_dir ./output \
    --skip_hpo true \
    -profile docker
```

### カスタムパラメータでの実行

```bash
nextflow run main.nf \
    --input samples_metadata.csv \
    --output_dir ./output \
    --model_name facebook/esm2_t33_650M_UR50D \
    --use_lora true \
    --lora_r 16 \
    --batch_size 16 \
    --epochs 30 \
    --n_trials 10 \
    --wandb_project my-esm2-project \
    -profile docker
```

---

## 設定

すべてのパラメータは `nextflow.config` で定義されており、コマンドラインフラグで上書きできます。

### データパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--seq_column` | `mutated_sequence` | タンパク質配列のカラム名 |
| `--max_length` | `405` | 最大配列長（切り詰め） |
| `--test_size` | `0.2` | テストセットの分割比率 |
| `--random_state` | `42` | 再現性のためのランダムシード |

### モデルパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--model_name` | `esm2_t36_3B_UR50D` | ESM-2 事前学習モデルのパスまたは HuggingFace ID |
| `--num_labels` | `1` | 出力次元（1 = 回帰） |
| `--freeze_layers` | `0` | 固定する下部エンコーダーレイヤー数 |

### LoRA パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--use_lora` | `true` | LoRA ファインチューニングを有効化 |
| `--lora_r` | `8` | LoRA ランク |
| `--lora_alpha` | `16` | LoRA アルファ（スケーリング係数） |
| `--lora_dropout` | `0.1` | LoRA ドロップアウト率 |
| `--target_modules` | `query,key,value,dense` | LoRA を適用する Attention モジュール |

### 学習パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch_size` | `8` | 学習バッチサイズ |
| `--gradient_accumulation_steps` | `2` | 勾配累積ステップ数 |
| `--epochs` | `20` | 最終学習エポック数 |
| `--learning_rate` | `5e-5` | 初期学習率 |
| `--optimizer` | `AdamW` | オプティマイザ（AdamW / Adam / SGD） |
| `--scheduler` | `CosineAnnealingLR` | 学習率スケジューラ |
| `--weight_decay` | `0.01` | Weight Decay |
| `--use_amp` | `true` | 混合精度を有効化 |
| `--patience` | `3` | アーリーストッピングの patience |
| `--min_delta` | `0.0001` | 改善とみなす最小変化量 |

### HPO パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--skip_hpo` | `false` | HPO ステージをスキップ |
| `--n_trials` | `5` | Optuna トライアル数 |
| `--epochs_hpo` | `5` | HPO 1 トライアルあたりのエポック数 |
| `--lr_search_range` | `5e-6 1e-4` | 学習率の探索範囲 |

### WandB パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--wandb_project` | `esm2-dms-HPO-...` | WandB プロジェクト名 |
| `--wandb_run_prefix` | `esm2` | 実行名のプレフィックス |

### リソースパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--gpus` | `1` | プロセスあたりの GPU 数 |
| `--cpus` | `104` | プロセスあたりの CPU 数 |
| `--memory` | `48.GB` | プロセスあたりのメモリ |
| `--maxForks` | `2` | 最大並列プロセス数 |

---

## 出力

各データセットについて、パイプラインは以下を生成します:

```
output_dir/
└── {dataset_id}/
    ├── final_model_weights/
    │   ├── adapter_config.json       # LoRA アダプター設定
    │   ├── adapter_model.safetensors # LoRA アダプター重み
    │   ├── scaler.joblib             # 目的変数の逆変換用 StandardScaler
    │   ├── tokenizer.json            # ESM-2 トークナイザー
    │   ├── train.csv                 # 学習用分割データ
    │   ├── val.csv                   # 検証用分割データ
    │   ├── test.csv                  # テスト用分割データ（予測に使用）
    │   ├── best_hpo_params.json      # 最適ハイパーパラメータ（HPO 有効時）
    │   └── wandb_run_id.txt          # 再開用 WandB 実行 ID
    └── predictions/
        └── {dataset_id}_predictions.csv  # 予測 DMS スコア
```

さらに Nextflow が `output_dir/reports/` 配下に実行レポートを生成します:
- `execution_report.html` — リソース使用状況とタスクサマリー
- `timeline.html` — プロセス実行タイムライン
- `dag.html` — ワークフロー DAG 可視化

---

## モデルアーキテクチャ

ファインチューニングヘッドは、ESM-2 のデフォルトプーリングを **CLS トークンプーリング** に置き換え、2 層 MLP を接続します:

```
ESM-2 エンコーダー → CLS トークン埋め込み → Linear(hidden, hidden/2) → ReLU → Dropout → Linear(hidden/2, 1)
```

- **損失関数**: StandardScaler で正規化された DMS スコアに対する MSE（平均二乗誤差）
- **評価指標**: 予測スコアと真のスコア間のスピアマン順位相関係数
- **目的変数の正規化**: 学習セットで `StandardScaler` を適合；推論時に逆変換を適用

---

## WandB 連携

各パイプラインステージは WandB に記録されます:

- **HPO ステージ**: 別プロジェクト（`{project}_HPO`）、トライアルごとのメトリクス、最適パラメータのサマリー
- **学習ステージ**: 損失曲線、検証スピアマン相関、学習率スケジュール、勾配ノルム、モデルアーティファクトのアップロード
- **推論ステージ**: 予測のヒストグラム、プレビューテーブル、入出力データセットのアーティファクト

実行名は `{prefix}-{model_short}-{lora|full}-{dataset_id}` のパターンに従います。

---

## Docker 環境

Docker イメージは以下を提供します:

- **ベース**: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
- **Python**: 3.13
- **PyTorch**: CUDA 12.8 対応 2.6.0
- **主要ライブラリ**: transformers 5.14、peft 0.20、accelerate 1.14、optuna 4.9、wandb 0.28、scikit-learn 1.9、pandas 3.0
- **パッケージマネージャ**: 高速で再現性のあるインストールのための `uv`

---

## トラブルシューティング

### CUDA Out of Memory
- `--batch_size` を減らすか、`--gradient_accumulation_steps` を増やしてください
- より小さい ESM-2 モデル（例: `esm2_t33_650M_UR50D`）を使用してください
- `--use_amp true` が設定されていることを確認してください

### WandB ログインに失敗する
- `WANDB_API_KEY` 環境変数が設定されているか確認してください
- コンテナ内で `wandb login` を実行し、接続性をテストしてください

### HPO トライアルがすべてプルーニングされる
- `--lr_search_range` を広げてください
- `--epochs_hpo` を増やし、学習シグナルを得られるようにしてください
- HPO 実行の `--patience` を小さくしてください

---

## ライセンス

本プロジェクトは研究目的で提供されています。ESM-2 は Facebook AI Research（FAIR）により、それぞれのライセンスの下でリリースされています。

---

## 引用

本パイプラインを研究で使用する場合は、以下を引用してください:

- ESM-2: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," *Science*, 2023.
- ProteinGym / DMS リファレンスデータセットについては、適宜該当する文献を引用してください。
