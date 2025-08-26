@echo off
echo ===================================
echo   CKD5期合并CAP生存预测Web应用启动脚本
echo ===================================
echo.
echo 正在检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境，请先安装Python 3.7+
    pause
    exit /b 1
)

echo.
echo 正在安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告: 依赖包安装可能存在问题
)

echo.
echo 启动Web应用...
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.
python app.py

pause