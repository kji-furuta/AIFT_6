# AI Fine-tuning Toolkit

🚀 **日本語LLMファインチューニング用の統合Webツールキット**

Dockerベースの統合Webインターフェースで、日本語大規模言語モデル（LLM）のファインチューニングを簡単に実行できます。フルファインチューニング、LoRA、QLoRAなど複数の手法をWebブラウザから直感的に操作可能です。

## 🌟 主要機能

### 🌐 統合Webインターフェース
- **ブラウザベースUI**: http://localhost:8050 でアクセス
- **リアルタイム監視**: ファインチューニング進捗の可視化
- **モデル管理**: 学習済みモデルの一覧・選択・生成
- **データアップロード**: JSONLファイルの簡単アップロード
- **システム情報**: GPU使用状況とメモリ監視
- **プロフェッショナルデザイン**: 帝国大学ロゴと洗練されたUI

### ファインチューニング手法
- **🔥 フルファインチューニング**: 全パラメータ更新による高精度学習
- **⚡ LoRA**: パラメータ効率的学習（低メモリ）
- **💎 QLoRA**: 4bit/8bit量子化による超省メモリ学習
- **🧠 EWC**: 継続的学習による破滅的忘却の抑制
- **🔧 自動量子化**: モデルサイズに応じた最適化

### ✅ サポートモデル
最新のサポートモデルリストです。

| モデル名 | タイプ | 精度 | 推奨VRAM | タグ |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen/Qwen2.5-14B-Instruct** | CausalLM | bfloat16 | 32GB | `multilingual`, `14b`, `instruct` |
| **Qwen/Qwen2.5-32B-Instruct** | CausalLM | bfloat16 | 80GB | `multilingual`, `32b`, `instruct` |
| **cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese** | CausalLM | bfloat16 | 80GB | `japanese`, `32b`, `deepseek` |
| **cyberagent/calm3-22b-chat** | CausalLM | float16 | 48GB | `japanese`, `22b`, `chat` |
| **meta-llama/Meta-Llama-3.1-70B-Instruct** | CausalLM | bfloat16 | 160GB | `multilingual`, `70b`, `instruct` |
| **meta-llama/Meta-Llama-3.1-32B-Instruct** | CausalLM | bfloat16 | 80GB | `multilingual`, `32b`, `instruct` |
| **microsoft/Phi-3.5-32B-Instruct** | CausalLM | bfloat16 | 80GB | `multilingual`, `32b`, `instruct` |
| **microsoft/Orca-2-32B-Instruct** | CausalLM | bfloat16 | 80GB | `multilingual`, `32b`, `instruct` |

### GPU最適化
- **Flash Attention 2**: 注意機構の高速化
- **Gradient Checkpointing**: メモリ使用量削減
- **Mixed Precision**: FP16による計算高速化
- **マルチGPU対応**: DataParallel/DistributedDataParallel

### 🧠 メモリ最適化（新機能）
- **動的量子化**: 32B/22Bモデルは4bit、7B/8Bモデルは8bit量子化を自動選択
- **CPUオフロード**: GPUメモリ不足時の自動CPU実行
- **メモリ監視**: リアルタイムメモリ使用量の監視と警告
- **モデルキャッシュ**: 効率的なモデル再利用
- **最適化されたAPI**: メモリ効率的なWeb API（`app/main_unified.py`）

## 📋 必要環境

### ハードウェア要件
- **GPU**: NVIDIA GPU（CUDA対応）
- **メモリ**: 最低8GB VRAM（推奨16GB以上）
- **システムメモリ**: 16GB以上推奨

### ソフトウェア要件
- Python 3.8以上（推奨3.11）
- CUDA 12.6+
- Docker & Docker Compose
- Git

## 🚀 クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/kji-furuta/AI_FT_3.git
cd AI_FT_3
```

### 2. Docker環境の起動
```bash
cd docker
docker-compose up -d --build
```

### 3. Webインターフェースの起動
```bash
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 4. ブラウザでアクセス
```
http://localhost:8050
```

### 🎯 使用可能な機能
- **ダッシュボード**: システム状況とタスク管理
- **ファインチューニング**: データアップロードと学習実行
- **テキスト生成**: 学習済みモデルでの推論
- **モデル管理**: 利用可能モデルと学習済みモデル一覧
- **マニュアル**: `/manual` - 詳細な利用方法
- **システム概要**: `/system-overview` - 技術仕様

## 📚 使用方法

### 🌐 Webインターフェース（推奨）

ブラウザで `http://localhost:8050` にアクセスして以下の機能を利用：

#### 1. ファインチューニング
1. **データアップロード**: JSONLファイルを選択・アップロード
2. **モデル選択**: 利用可能なベースモデルから選択
3. **設定調整**: LoRA/QLoRA/フルファインチューニングの選択
4. **実行監視**: リアルタイム進捗とログの確認

#### 2. テキスト生成
1. **モデル選択**: 学習済みモデルの選択
2. **プロンプト入力**: 生成したいテキストの入力
3. **パラメータ調整**: 温度、最大長などの設定
4. **結果確認**: 生成されたテキストの表示・保存

#### 3. システム管理
- **システム情報**: GPU使用状況とメモリ監視
- **モデル一覧**: 利用可能・学習済みモデルの管理
- **ドキュメント**: マニュアルと技術仕様の参照

### 🔧 API使用（上級者向け）

### LoRAファインチューニングの例
```python
from src.models.japanese_model import JapaneseModel
from src.training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
from src.training.training_utils import TrainingConfig

# モデルの初期化 (新しい推奨モデル)
model = JapaneseModel(
    model_name="cyberagent/calm3-22b-chat"  # または "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
)

# LoRA設定
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    use_qlora=False
)

# トレーニング設定
training_config = TrainingConfig(
    learning_rate=3e-4,
    batch_size=4,
    num_epochs=3,
    output_dir="./outputs/lora_stablelm_3b"
)

# トレーナーの初期化と実行
trainer = LoRAFinetuningTrainer(model, lora_config, training_config)
trainer.train(train_texts=["日本の首都は東京です。", "日本の最高峰は富士山です。"])
```

### 🧠 EWCによる継続的学習の例
EWCは、以前のタスクの知識を忘れることなく、新しいタスクをモデルに学習させるための手法です。

```python
from src.models.japanese_model import JapaneseModel
from src.training.lora_finetuning import LoRAFinetuningTrainer, LoRAConfig
from src.training.training_utils import TrainingConfig
from src.training.ewc_utils import EWCConfig, EWCManager

# 1. ベースモデルと最初のタスクのデータ
model = JapaneseModel("cyberagent/calm3-22b-chat")
task1_data = ["一般的な知識に関するテキスト...", "歴史に関するテキスト..."]

# 2. 最初のタスクでモデルをファインチューニング
lora_config = LoRAConfig(r=8, lora_alpha=16)
training_config = TrainingConfig(learning_rate=2e-4, num_epochs=2, output_dir="./outputs/task1_lora")
trainer = LoRAFinetuningTrainer(model, lora_config, training_config)
trained_model = trainer.train(train_texts=task1_data)

# 3. EWCの準備 (Fisher情報行列の計算)
ewc_manager = EWCManager(trained_model.model, trained_model.tokenizer)
fisher_matrix = ewc_manager.compute_fisher(task1_data)

# 4. 新しいタスクでEWCを使ってファインチューニング
ewc_config = EWCConfig(enabled=True, ewc_lambda=0.5, fisher_matrix=fisher_matrix)
training_config_task2 = TrainingConfig(learning_rate=1e-4, num_epochs=2, output_dir="./outputs/task2_ewc_lora")

task2_data = ["プログラミングに関するテキスト...", "Pythonのコード例..."]
trainer_task2 = LoRAFinetuningTrainer(
    model=trained_model, 
    lora_config=lora_config, 
    training_config=training_config_task2,
    ewc_config=ewc_config # EWC設定を渡す
)
final_model = trainer_task2.train(train_texts=task2_data)
```

## 📁 プロジェクト構造

```
AI_FT_3/
├── app/                          # Webアプリケーション
│   ├── main_unified.py           # 統合Webサーバー（稼働中）
│   ├── memory_optimized_loader.py # メモリ最適化ローダー
│   └── static/                   # フロントエンドファイル
│       └── logo_teikoku.png      # 帝国大学ロゴ
├── templates/                    # HTMLテンプレート
│   ├── base.html                 # ベーステンプレート（ロゴ統合）
│   ├── index.html                # メインページ
│   ├── finetune.html             # ファインチューニングページ
│   └── models.html               # モデル管理ページ
├── static/                       # 静的ファイル（templatesと同じレベル）
│   └── logo_teikoku.png          # ロゴファイル（Web配信用）
├── src/                          # コアライブラリ
│   ├── models/                   # モデル関連
│   ├── training/                 # ファインチューニング
│   ├── utils/                    # ユーティリティ
│   └── rag/                      # RAG機能
├── docker/                       # Docker環境
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/                      # 運用スクリプト
├── outputs/                      # 学習済みモデル保存
├── data/                         # トレーニングデータ
├── config/                       # 基本設定
├── configs/                      # DeepSpeed設定
└── docs/                         # ドキュメント
    ├── API_REFERENCE.md
    ├── LARGE_MODEL_SETUP.md
    └── MULTI_GPU_OPTIMIZATION.md
```

## ✨ 主な特徴

### 🎯 簡単操作
- **ワンクリック起動**: Docker Composeで環境構築完了
- **ブラウザ操作**: プログラミング不要のWebUI
- **リアルタイム監視**: 学習進捗とGPU使用状況を可視化
- **自動最適化**: モデルサイズに応じた量子化設定
- **プロフェッショナルUI**: 帝国大学ロゴと洗練されたデザイン

### 🚀 高性能
- **GPU最適化**: CUDA 12.6 + PyTorch 2.7.1
- **メモリ効率**: 動的量子化とキャッシュ管理
- **マルチモデル対応**: 3B〜70Bモデルまでサポート
- **DeepSpeed対応**: 将来の大規模学習に対応
- **静的ファイル最適化**: 統合されたディレクトリ構造

### 🎨 UI/UX改善
- **ロゴ統合**: 株）テイコク　ロゴ（300px × 150px）の表示
- **レスポンシブデザイン**: 様々な画面サイズに対応
- **ダークテーマ**: 濃い背景色と薄い文字色で視認性向上
- **コンパクトレイアウト**: 効率的なスペース利用

## 🤝 コントリビューション

プルリクエストを歓迎します。主な開発ブランチは `main` です。

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/new-feature`)
3. 変更をコミット (`git commit -m 'Add new feature'`)
4. ブランチにプッシュ (`git push origin feature/new-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Accelerate](https://github.com/huggingface/accelerate)

## 📚 関連ドキュメント

- [API リファレンス](docs/API_REFERENCE.md) - 詳細なAPI仕様
- [大規模モデルセットアップ](docs/LARGE_MODEL_SETUP.md) - 32B+モデルの設定方法
- [マルチGPU最適化](docs/MULTI_GPU_OPTIMIZATION.md) - 分散学習の設定

### 🌐 Webドキュメント
- **利用マニュアル**: http://localhost:8050/manual
- **システム概要**: http://localhost:8050/system-overview

---

## 🎯 今すぐ始める

```bash
# 1. クローン
git clone https://github.com/kji-furuta/AI_FT_3.git
cd AI_FT_3

# 2. 起動
cd docker && docker-compose up -d --build

# 3. Webサーバー開始
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh

# 4. ブラウザでアクセス
# http://localhost:8050
```

**🚀 5分でファインチューニング開始！**

### 🔧 トラブルシューティング

#### ロゴが表示されない場合
```bash
# 静的ファイルの確認
docker exec ai-ft-container ls -la /workspace/static/

# ロゴファイルの存在確認
docker exec ai-ft-container curl -I http://localhost:8050/static/logo_teikoku.png
```

#### Webインターフェースが起動しない場合
```bash
# コンテナの状態確認
docker ps -a

# ログの確認
docker logs ai-ft-container

# 手動起動
docker exec -d ai-ft-container python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```
