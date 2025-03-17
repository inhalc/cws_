# 中文分词模型处理、训练与测试Makefile

# 变量定义
PYTHON = python
DATA_DIR = data/sighan2005
MODEL_DIR = models
TRAIN_SCRIPT = src/train.py
TEST_SCRIPT = tests/test_features.py
PREPROCESS_SCRIPT = data/preprocessing.py

# 数据集文件
MSR_TRAIN = $(DATA_DIR)/msr_training_words.utf8
MSR_TEST = $(DATA_DIR)/msr_test_gold.utf8
PKU_TRAIN = $(DATA_DIR)/pku_training_words.utf8
PKU_TEST = $(DATA_DIR)/pku_test_gold.utf8

# 模型文件
MSR_MODEL = $(MODEL_DIR)/msr_perceptron.pkl
PKU_MODEL = $(MODEL_DIR)/pku_perceptron.pkl

# 默认目标
.PHONY: all
all: prepare train test

# 创建必要的目录
.PHONY: prepare
prepare:
    @echo "创建必要的目录..."
    mkdir -p $(DATA_DIR)/training
    mkdir -p $(DATA_DIR)/gold
    mkdir -p $(MODEL_DIR)
    @echo "目录创建完成"

# 处理数据集
.PHONY: preprocess
preprocess: prepare
    @echo "处理SIGHAN2005数据集..."
    $(PYTHON) $(PREPROCESS_SCRIPT) --data_dir ./$(DATA_DIR) --output_dir ./$(DATA_DIR)
    @echo "数据处理完成"

# 训练MSR模型
.PHONY: train-msr
train-msr: $(MSR_TRAIN)
    @echo "开始训练MSR模型..."
    $(PYTHON) $(TRAIN_SCRIPT) --train_file $(MSR_TRAIN) --model_file $(MSR_MODEL) --epochs 5
    @echo "MSR模型训练完成"

# 训练PKU模型
.PHONY: train-pku
train-pku: $(PKU_TRAIN)
    @echo "开始训练PKU模型..."
    $(PYTHON) $(TRAIN_SCRIPT) --train_file $(PKU_TRAIN) --model_file $(PKU_MODEL) --epochs 5
    @echo "PKU模型训练完成"

# 训练所有模型
.PHONY: train
train: train-msr train-pku

# 测试MSR模型
.PHONY: test-msr
test-msr: $(MSR_MODEL) $(MSR_TEST)
    @echo "开始测试MSR模型..."
    $(PYTHON) $(TEST_SCRIPT) --model_file $(MSR_MODEL) --test_file $(MSR_TEST)
    @echo "MSR模型测试完成"

# 测试PKU模型
.PHONY: test-pku
test-pku: $(PKU_MODEL) $(PKU_TEST)
    @echo "开始测试PKU模型..."
    $(PYTHON) $(TEST_SCRIPT) --model_file $(PKU_MODEL) --test_file $(PKU_TEST)
    @echo "PKU模型测试完成"

# 测试所有模型
.PHONY: test
test: test-msr test-pku

# 下载SIGHAN2005数据集
.PHONY: download-data
download-data:
    @echo "下载SIGHAN2005数据集..."
    mkdir -p temp
    curl -L http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip -o temp/icwb2-data.zip
    unzip -o temp/icwb2-data.zip -d temp
    mkdir -p $(DATA_DIR)/training
    mkdir -p $(DATA_DIR)/gold
    cp temp/icwb2-data/training/msr_training.utf8 $(DATA_DIR)/training/
    cp temp/icwb2-data/training/pku_training.utf8 $(DATA_DIR)/training/
    cp temp/icwb2-data/gold/msr_test_gold.utf8 $(DATA_DIR)/gold/
    cp temp/icwb2-data/gold/pku_test_gold.utf8 $(DATA_DIR)/gold/
    rm -rf temp
    @echo "数据下载完成"

# 创建配置文件
.PHONY: config
config:
    @echo "创建配置文件..."
    @echo "# 中文分词模型配置" > config.yaml
    @echo "data:" >> config.yaml
    @echo "  train_file: $(MSR_TRAIN)" >> config.yaml
    @echo "  test_file: $(MSR_TEST)" >> config.yaml
    @echo "model:" >> config.yaml
    @echo "  epochs: 5" >> config.yaml
    @echo "  model_file: $(MSR_MODEL)" >> config.yaml
    @echo "评估:" >> config.yaml
    @echo "  verbose: true" >> config.yaml
    @echo "配置文件已创建: config.yaml"

# 清理生成的文件
.PHONY: clean
clean:
    @echo "清理生成的文件..."
    rm -f $(MSR_MODEL) $(PKU_MODEL)
    @echo "清理完成"

# 完全清理（包括数据集）
.PHONY: clean-all
clean-all: clean
    @echo "清理所有生成的文件和数据..."
    rm -rf $(DATA_DIR)
    rm -rf $(MODEL_DIR)
    @echo "完全清理完成"

# 安装依赖
.PHONY: install
install:
    @echo "安装项目依赖..."
    pip install numpy tqdm scikit-learn pyyaml
    @echo "依赖安装完成"

# 显示帮助信息
.PHONY: help
help:
    @echo "中文分词模型处理、训练与测试命令："
    @echo ""
    @echo "make prepare        - 创建必要的目录结构"
    @echo "make download-data  - 下载SIGHAN2005数据集"
    @echo "make preprocess     - 处理数据集"
    @echo "make train          - 训练所有模型"
    @echo "make train-msr      - 仅训练MSR模型"
    @echo "make train-pku      - 仅训练PKU模型"
    @echo "make test           - 测试所有模型"
    @echo "make test-msr       - 仅测试MSR模型"
    @echo "make test-pku       - 仅测试PKU模型" 
    @echo "make config         - 创建配置文件"
    @echo "make clean          - 清理模型文件"
    @echo "make clean-all      - 清理所有生成的文件和数据"
    @echo "make install        - 安装项目依赖"
    @echo "make all            - 执行完整流程（准备、训练、测试）"
    @echo ""