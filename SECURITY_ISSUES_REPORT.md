# 安全问题分析报告

## 项目概述
项目名称：CKD5期合并CAP生存预测Web应用  
分析日期：2025-01-15  
分析范围：代码安全性、配置安全性、依赖项安全性

## 🔴 高风险问题

### 1. Flask应用网络配置安全风险
**问题描述：**
- 应用配置为监听所有网络接口 (`host='0.0.0.0'`)
- 这使得应用可以从任何网络接口访问，包括公网

**风险等级：** 🔴 高风险

**影响：**
- 如果服务器有公网IP，应用将暴露在互联网上
- 可能遭受未授权访问和攻击

**建议修复：**
```python
# 生产环境建议
if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)  # 仅本地访问
    
# 或者使用环境变量控制
import os
host = os.getenv('FLASK_HOST', '127.0.0.1')
app.run(debug=False, host=host, port=5000)
```

### 2. 敏感信息硬编码
**问题描述：**
- Flask secret_key 直接硬编码在源代码中
- `app.secret_key = 'ckd5_cap_survival_prediction_app_2025'`

**风险等级：** 🔴 高风险

**影响：**
- session可能被伪造
- 如果代码泄露，攻击者可以伪造用户会话

**建议修复：**
```python
import os
import secrets

# 使用环境变量
app.secret_key = os.getenv('FLASK_SECRET_KEY') or secrets.token_hex(32)

# 或者从配置文件读取
# app.secret_key = load_secret_from_config()
```

### 3. 文件上传安全漏洞
**问题描述：**
- 批量预测功能缺乏文件大小限制
- 没有文件内容验证
- 可能导致拒绝服务攻击或恶意文件上传

**风险等级：** 🔴 高风险

**影响：**
- 大文件上传可能耗尽服务器资源
- 恶意构造的文件可能导致应用崩溃

**建议修复：**
```python
# 在app.py中添加配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 在batch_predict函数中添加验证
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未找到上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
            
        # 验证文件扩展名
        allowed_extensions = {'.csv', '.xlsx', '.xls'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': '不支持的文件格式'}), 400
            
        # 验证文件大小
        if len(file.read()) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': '文件过大'}), 400
        file.seek(0)  # 重置文件指针
        
        # 限制处理行数
        MAX_ROWS = 1000
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, nrows=MAX_ROWS)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file, nrows=MAX_ROWS)
            
        # ... 其余代码
```

## 🟡 中等风险问题

### 4. 依赖项版本管理
**问题描述：**
- requirements.txt使用 `>=` 版本约束
- 可能引入不兼容或有安全漏洞的新版本

**风险等级：** 🟡 中等风险

**建议修复：**
```txt
# 建议使用固定版本或兼容版本范围
flask==2.3.3
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
xgboost==1.7.6
shap==0.42.1
matplotlib==3.7.2
seaborn==0.12.2
werkzeug==2.3.7
```

### 5. 错误信息泄露
**问题描述：**
- 异常处理中直接返回 `str(e)`
- 可能泄露系统内部信息

**风险等级：** 🟡 中等风险

**建议修复：**
```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 在异常处理中
try:
    # ... 业务逻辑
except Exception as e:
    logger.error(f"预测失败: {str(e)}", exc_info=True)
    return jsonify({'error': '预测服务暂时不可用，请稍后重试'}), 500
```

## 🟢 改进建议

### 1. 添加输入验证
```python
def validate_input_data(data):
    """验证输入数据"""
    required_fields = ['CURB65评分', '重度肺炎', 'DDimer', '炎症累及肺炎数≥3', 
                      'PLR', '年龄', '炎症累计双肺', '冠心病', '饮酒史', 'UA']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"缺少必需字段: {field}")
            
    # 数值范围验证
    if not (0 <= data.get('CURB65评分', 0) <= 5):
        raise ValueError("CURB65评分必须在0-5之间")
        
    if not (0 <= data.get('年龄', 0) <= 120):
        raise ValueError("年龄必须在合理范围内")
        
    # ... 其他验证逻辑
```

### 2. 添加访问控制
```python
from functools import wraps
from flask import request, abort

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            abort(401)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... 预测逻辑
```

### 3. 添加速率限制
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... 预测逻辑
```

## 总结

该项目存在多个安全风险，主要集中在：
1. **网络安全配置** - 需要限制访问范围
2. **敏感信息管理** - 需要使用环境变量管理密钥
3. **文件上传安全** - 需要添加验证和限制
4. **依赖项管理** - 需要固定版本避免安全漏洞
5. **错误处理** - 需要避免信息泄露

建议按优先级逐步修复这些问题，特别是高风险问题应立即处理。

## 下一步行动

1. 立即修复高风险问题
2. 实施输入验证和访问控制
3. 添加日志记录和监控
4. 定期更新依赖项并进行安全扫描
5. 考虑使用HTTPS和其他安全传输措施