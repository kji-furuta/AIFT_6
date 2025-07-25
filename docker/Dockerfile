# AI Fine-Tuning Dockerfile
# AIファインチューニング用のDockerfile

# PyTorchの公式イメージをベースイメージとして使用（CUDA 12.6、cuDNN 9対応の開発版）
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# 作業ディレクトリを/workspaceに設定
WORKDIR /workspace

# 環境変数の設定
# Pythonパスをワークスペースディレクトリに設定
ENV PYTHONPATH=/workspace
# Debianパッケージのインストール時に対話型プロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive
# 使用するGPUデバイスを0番と1番に指定
ENV CUDA_VISIBLE_DEVICES=0,1
# 環境変数でトークンを設定
ENV JUPYTER_TOKEN=${JUPYTER_TOKEN:-"your-secure-token"}

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 高速パッケージ管理ツールuvをインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# uvのパスを環境変数PATHに追加
ENV PATH="/root/.local/bin:$PATH"

# 効率的なキャッシュのために、要件ファイルを先にコピー
COPY requirements.txt /workspace/
COPY pyproject.toml /workspace/

# uvを使用してrequirements.txtからPythonパッケージをインストール
RUN uv pip install --system -r requirements.txt

# flash-attn（高速アテンション実装）をtorchと他の依存関係インストール後に個別インストール
RUN uv pip install --system flash-attn>=2.0.0 --no-build-isolation

# 開発用およびWebインターフェース用の追加パッケージをインストール
RUN uv pip install --system \
    ipython \
    jupyter \
    jupyterlab \
    tensorboard \
    wandb \
    matplotlib \
    seaborn \
    plotly \
    fastapi \
    uvicorn \
    python-multipart \
    psutil

# 必要なディレクトリを作成
RUN mkdir -p /workspace/data/raw \
    /workspace/data/processed \
    /workspace/data/rag_documents \
    /workspace/data/uploaded \
    /workspace/models/checkpoints \
    /workspace/logs \
    /workspace/outputs \
    /workspace/app/static

# 非rootユーザーの作成（セキュリティ向上のため）
RUN useradd -m -s /bin/bash -u 1000 ai-user

# 必要なファイルのみをコピー（.dockerignoreで除外されるファイルは除く）
COPY src/ /workspace/src/
COPY config/ /workspace/config/
COPY scripts/ /workspace/scripts/
COPY notebooks/ /workspace/notebooks/
COPY tests/ /workspace/tests/
COPY app/ /workspace/app/
COPY templates/ /workspace/templates/
COPY docs/ /workspace/docs/
COPY examples/ /workspace/examples/
COPY *.py /workspace/
COPY *.json /workspace/
COPY *.md /workspace/
COPY *.sh /workspace/
COPY *.bat /workspace/

# 権限を設定（所有者変更を最適化）
RUN chown -R ai-user:ai-user /workspace && \
    chmod -R 755 /workspace && \
    chmod -R 777 /workspace/data/uploaded /workspace/logs /workspace/models /workspace/data

# JupyterLabの設定
RUN jupyter lab --generate-config && \
    mkdir -p /home/ai-user/.jupyter && \
    cp /root/.jupyter/jupyter_lab_config.py /home/ai-user/.jupyter/ && \
    chown -R ai-user:ai-user /home/ai-user/.jupyter

# JupyterLab設定を追加
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/ai-user/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /home/ai-user/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

# GPUアクセスとPyTorchインストールの検証
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 作業ユーザーを切り替え
USER ai-user

# ポートを公開
EXPOSE 8888 6006 8050

# デフォルトコマンド（bashシェルを起動）
CMD ["/bin/bash"]