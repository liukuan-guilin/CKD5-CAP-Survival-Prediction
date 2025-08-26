#!/bin/bash

echo "==================================="
echo "  CKD5期合并CAP生存预测Web应用启动脚本"
echo "==================================="
echo

echo "正在检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3环境，请先安装Python 3.7+"
    exit 1
fi

python3 --version
echo

echo "正在安装依赖包..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "警告: 依赖包安装可能存在问题"
fi

echo
echo "启动Web应用..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo

python3 app.py