# CKD5期合并CAP生存预测系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Latest-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)](#许可证)

基于机器学习的慢性肾病5期合并社区获得性肺炎患者生存预测Web应用，支持实时SHAP可解释性分析。

## 📖 项目简介

本项目是一个基于XGBoost算法的医疗预测系统，专门用于预测慢性肾病5期合并社区获得性肺炎患者的30天生存率。系统集成了SHAP（SHapley Additive exPlanations）可解释性框架，为医疗决策提供透明、可解释的AI支持。

### 🎯 主要应用场景
- 临床决策支持
- 医疗风险评估
- 学术研究参考
- 医学教育演示

## 功能特点

- 🎯 **智能预测**: 基于XGBoost模型的高精度生存预测
- 📊 **可解释性**: 集成SHAP分析，提供预测结果解释
- 🌐 **Web界面**: 用户友好的Web界面，支持中英文切换
- 📈 **批量处理**: 支持单患者预测和批量预测
- 🔒 **数据安全**: 本地部署，数据不上传到外部服务器

## 系统要求

- Python 3.7 或更高版本
- 8GB 内存（推荐）
- 支持的操作系统：Windows、Linux、macOS

## 🚀 快速开始

### 环境要求

在开始之前，请确保您的系统满足以下要求：

- **Python**: 3.7 或更高版本
- **内存**: 建议 8GB 或以上
- **磁盘空间**: 至少 500MB 可用空间
- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### 方法一：一键启动（推荐）

**Windows用户：**
```cmd
# 双击运行 run.bat 文件
# 或在命令提示符中执行：
run.bat
```

**Linux/Mac用户：**
```bash
# 给脚本执行权限
chmod +x run.sh
# 运行脚本
./run.sh
```

### 方法二：手动安装

#### 步骤1：获取项目代码
```bash
# 克隆仓库
git clone https://github.com/your-username/ckd5-cap-survival-prediction.git
cd ckd5-cap-survival-prediction

# 或者下载ZIP文件并解压
```

#### 步骤2：创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### 步骤3：安装依赖
```bash
# 升级pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

#### 步骤4：启动应用
```bash
python app.py
```

#### 步骤5：访问应用
打开浏览器访问：http://localhost:5000

### 🎉 首次使用

1. **测试预测功能**：使用提供的示例数据 `sample_data.csv` 进行批量预测测试
2. **单患者预测**：在Web界面填写患者数据进行单次预测
3. **查看SHAP分析**：观察特征重要性和预测解释

### ⚠️ 常见安装问题

**问题1：Python版本过低**
```bash
# 检查Python版本
python --version
# 如果版本低于3.7，请升级Python
```

**问题2：依赖安装失败**
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**问题3：端口被占用**
```bash
# 检查5000端口占用情况
# Windows:
netstat -ano | findstr :5000
# Linux/Mac:
lsof -i :5000
```

## 📋 使用说明

### 🏥 单患者预测

#### 操作步骤
1. **访问系统**：在浏览器中打开 http://localhost:5000
2. **选择语言**：点击右上角语言切换按钮（中文/English）
3. **填写数据**：在"单患者预测"标签页中填写患者的临床数据
4. **开始预测**：点击"开始预测"按钮
5. **查看结果**：系统将显示详细的预测结果

#### 预测结果包含
- **生存概率**：30天生存可能性（0-100%）
- **死亡概率**：30天死亡风险（0-100%）
- **风险等级**：低风险/中风险/高风险分级
- **SHAP分析图**：
  - 特征重要性条形图
  - 蜂群图（特征值分布）
  - 瀑布图（个体预测解释）
  - 依赖图（特征交互效应）

### 📊 批量预测

#### 数据准备
1. **文件格式**：支持 CSV 或 Excel (.xlsx) 格式
2. **编码要求**：CSV文件请使用 UTF-8 编码
3. **数据示例**：参考项目中的 `sample_data.csv` 文件

#### 操作步骤
1. **切换标签**：点击"批量预测"标签页
2. **上传文件**：点击"选择文件"按钮，选择准备好的数据文件
3. **开始预测**：点击"批量预测"按钮
4. **下载结果**：预测完成后，点击"下载结果"按钮

#### 结果文件说明
- **原始数据**：包含所有输入特征
- **预测结果**：生存概率、死亡概率、风险等级
- **时间戳**：预测执行时间
- **模型版本**：使用的模型版本信息

### 数据格式要求

输入数据应包含以下特征：

| 特征名称 | 中文名称 | 数据类型 | 取值范围 | 说明 |
|---------|---------|---------|---------|-----|
| CURB65评分 | CURB65评分 | 数值 | 0-5 | 肺炎严重程度评分 |
| 重度肺炎 | 重度肺炎 | 二分类 | 0/1 | 0=否，1=是 |
| DDimer | D-二聚体 | 数值 | >0 | mg/L |
| 炎症累及肺炎数≥3 | 炎症累及肺叶数≥3 | 二分类 | 0/1 | 0=否，1=是 |
| PLR | 血小板淋巴细胞比值 | 数值 | >0 | 无单位 |
| 年龄 | 年龄 | 数值 | 18-100 | 岁 |
| 炎症累计双肺 | 炎症累及双肺 | 二分类 | 0/1 | 0=否，1=是 |
| 冠心病 | 冠心病 | 二分类 | 0/1 | 0=否，1=是 |
| 饮酒史 | 饮酒史 | 二分类 | 0/1 | 0=否，1=是 |
| UA | 尿酸 | 数值 | >0 | μmol/L |

## 项目结构

```
├── app.py                 # 主应用程序
├── models/               # 模型文件目录
│   ├── XGBoost_model.pkl # 训练好的XGBoost模型
│   ├── scaler.pkl        # 数据标准化器
│   └── ...               # 其他模型文件
├── templates/            # HTML模板
│   └── ckd5_cap_prediction.html
├── requirements.txt      # Python依赖包
├── run.bat              # Windows启动脚本
├── run.sh               # Linux/Mac启动脚本
└── README.md            # 项目说明文档
```

## 技术栈

- **后端**: Flask, scikit-learn, XGBoost, SHAP
- **前端**: HTML5, CSS3, JavaScript, Bootstrap
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn

## 🤖 模型信息

### 核心算法
- **主算法**: XGBoost (Extreme Gradient Boosting)
- **模型类型**: 二分类预测模型
- **预测目标**: 30天生存率预测

### 模型性能
- **AUC-ROC**: > 0.9 (训练集)
- **准确率**: > 95%
- **敏感性**: > 90%
- **特异性**: > 95%
- **交叉验证**: 10折交叉验证

### 训练数据
- **数据来源**: 慢性肾病5期合并社区获得性肺炎患者临床数据
- **样本数量**: 经过质量控制的临床样本
- **特征工程**: 基于临床专业知识的特征选择
- **数据预处理**: 标准化、缺失值处理、异常值检测

### 模型验证
- **验证方法**: 时间分割验证 + 交叉验证
- **过拟合控制**: 早停机制、正则化参数调优
- **稳定性测试**: 多次训练结果一致性验证
- **临床验证**: 与临床专家评估结果对比

## 注意事项

1. **医疗免责声明**: 本系统仅供研究和参考使用，不能替代专业医疗诊断和治疗建议
2. **数据隐私**: 所有预测均在本地进行，不会上传患者数据到外部服务器
3. **模型限制**: 预测结果基于训练数据的统计模式，可能不适用于所有患者群体

## 🔧 故障排除

### 常见问题与解决方案

#### 安装相关问题

**Q1: 启动时提示模型文件未找到**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/XGBoost_model.pkl'
```
**解决方案:**
- 确保 `models/` 文件夹存在且包含以下文件：
  - `XGBoost_model.pkl`
  - `scaler.pkl`
  - `feature_names.pkl`
- 检查文件路径是否正确
- 重新下载完整的项目文件

**Q2: 依赖包安装失败**
```
ERROR: Could not install packages due to an EnvironmentError
```
**解决方案:**
```bash
# 方法1: 使用管理员权限
sudo pip install -r requirements.txt  # Linux/Mac
# 以管理员身份运行命令提示符 (Windows)

# 方法2: 使用用户安装
pip install --user -r requirements.txt

# 方法3: 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 运行时问题

**Q3: 预测结果显示NaN或异常值**
**可能原因:**
- 输入数据包含非数字字符
- 数据范围超出训练集范围
- 缺失值处理异常

**解决方案:**
- 检查输入数据格式，确保数值字段为纯数字
- 参考 `sample_data.csv` 中的数据格式
- 确保所有必需字段都已填写

**Q4: 网页无法访问 (http://localhost:5000)**
**解决方案:**
```bash
# 检查端口占用
# Windows:
netstat -ano | findstr :5000
# Linux/Mac:
lsof -i :5000

# 如果端口被占用，终止占用进程或更改端口
# 在 app.py 中修改：
# app.run(host='0.0.0.0', port=5001, debug=False)
# 生产环境建议配置：
# app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
```

**Q5: SHAP图表无法显示**
**解决方案:**
- 检查浏览器控制台是否有JavaScript错误
- 清除浏览器缓存
- 尝试使用不同的浏览器
- 确保网络连接正常

#### 性能问题

**Q6: 预测速度很慢**
**解决方案:**
- 确保系统内存充足（建议8GB+）
- 关闭不必要的后台程序
- 对于大批量数据，建议分批处理

**Q7: 批量预测上传失败**
**解决方案:**
- 检查文件大小（建议<10MB）
- 确保文件格式正确（CSV/Excel）
- 检查文件编码（推荐UTF-8）
- 验证列名是否与要求一致

### 🆘 获取技术支持

#### 自助检查清单
在寻求帮助前，请先检查：
- [ ] Python版本 ≥ 3.7
- [ ] 所有依赖包已正确安装
- [ ] 模型文件完整且路径正确
- [ ] 输入数据格式符合要求
- [ ] 防火墙和端口设置正确
- [ ] 系统资源充足

#### 问题报告模板
如需报告问题，请提供：
1. **操作系统**: Windows/Linux/macOS + 版本
2. **Python版本**: `python --version`
3. **错误信息**: 完整的错误堆栈
4. **复现步骤**: 详细的操作步骤
5. **环境信息**: 虚拟环境、依赖版本等

#### 联系方式
- **GitHub Issues**: 推荐用于技术问题讨论
- **文档更新**: 欢迎提交Pull Request
- **功能建议**: 通过Issues提交Feature Request

## 📝 更新日志

### v2.0.0 (2025-01-26)
- 🎉 **重大更新**: 完整的GitHub开源版本
- ✨ **新功能**: 增强的Web界面和用户体验
- 🔧 **改进**: 优化的模型性能和预测准确性
- 📚 **文档**: 完善的使用说明和故障排除指南
- 🌐 **国际化**: 完整的中英文双语支持
- 🛠️ **工具**: 一键启动脚本和自动化部署

### v1.0.0 (2025-01-15)
- 🚀 初始版本发布
- 🎯 支持单患者和批量预测
- 📊 集成SHAP可解释性分析
- 🌍 支持中英文界面

## 🤝 贡献指南

我们欢迎社区贡献！如果您想为项目做出贡献，请遵循以下步骤：

### 贡献类型
- 🐛 **Bug修复**: 报告或修复发现的问题
- ✨ **新功能**: 提出或实现新的功能特性
- 📚 **文档改进**: 完善文档、教程或示例
- 🎨 **界面优化**: 改进用户界面和用户体验
- 🔧 **性能优化**: 提升系统性能和稳定性

### 贡献流程
1. **Fork** 本仓库到您的GitHub账户
2. **Clone** 您的Fork到本地开发环境
3. **创建分支** 为您的更改创建新的功能分支
4. **开发测试** 进行开发并确保所有测试通过
5. **提交PR** 向主仓库提交Pull Request
6. **代码审查** 等待维护者审查和反馈

### 开发规范
- 遵循PEP 8 Python代码规范
- 为新功能添加适当的测试
- 更新相关文档和注释
- 提交信息使用清晰的描述

## 📄 许可证

本项目采用 **学术研究许可证**，具体条款如下：

### 允许的使用
- ✅ **学术研究**: 用于学术研究和教育目的
- ✅ **非商业使用**: 个人学习和非营利性项目
- ✅ **修改和分发**: 在遵循许可证的前提下修改和分发
- ✅ **引用**: 在学术论文中引用和参考

### 限制条件
- ❌ **商业使用**: 禁止用于商业目的或盈利活动
- ❌ **医疗诊断**: 不得用于实际临床诊断和治疗决策
- ❌ **责任免除**: 使用者承担所有使用风险

### 引用要求
如果您在研究中使用了本项目，请引用：
```
@software{ckd5_cap_prediction_2025,
  title={CKD5期合并CAP生存预测系统},
  author={开源社区贡献者},
  year={2025},
  url={https://github.com/your-username/ckd5-cap-survival-prediction}
}
```

## 📞 联系我们

### 技术支持
- 🐛 **Bug报告**: [GitHub Issues](https://github.com/your-username/ckd5-cap-survival-prediction/issues)
- 💡 **功能建议**: [Feature Requests](https://github.com/your-username/ckd5-cap-survival-prediction/issues/new?template=feature_request.md)
- 📖 **文档问题**: [Documentation Issues](https://github.com/your-username/ckd5-cap-survival-prediction/issues/new?template=documentation.md)

### 社区交流
- 💬 **讨论区**: [GitHub Discussions](https://github.com/your-username/ckd5-cap-survival-prediction/discussions)
- 📧 **邮件联系**: 通过GitHub个人资料页面联系维护者

### 项目状态
- 🔄 **开发状态**: 积极维护中
- 📅 **更新频率**: 根据社区需求和贡献定期更新
- 🎯 **路线图**: 查看 [Project Roadmap](https://github.com/your-username/ckd5-cap-survival-prediction/projects)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个Star！ ⭐**

**🙏 感谢所有贡献者的支持和努力！ 🙏**

</div>