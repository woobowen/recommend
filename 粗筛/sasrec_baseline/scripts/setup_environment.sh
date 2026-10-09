#!/bin/bash

# SASRec环境设置脚本
# 用于设置Python路径和环境变量

set -e

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

echo "设置SASRec环境"
echo "SASRec项目根目录: $PROJECT_ROOT"
echo "原始项目根目录: $ORIGINAL_ROOT"

# 设置Python路径
export PYTHONPATH="$ORIGINAL_ROOT:$PROJECT_ROOT:$PYTHONPATH"

echo "Python路径已设置: $PYTHONPATH"

# 创建必要的目录
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/outputs"

echo "环境设置完成!"
echo ""
echo "使用方法:"
echo "source scripts/setup_environment.sh"
echo "然后运行训练或推理脚本"
