# GitHub手动上传指南

## 📋 准备工作检查清单

### ✅ 已完成的准备工作
- [x] 项目代码已完成
- [x] 安全问题已修复（Flask Secret Key、网络配置、依赖版本）
- [x] 本地Git仓库已初始化
- [x] 代码已提交到本地Git仓库

### 📝 需要准备的信息
- GitHub用户名
- 要创建的仓库名称（建议：`CKD5-CAP-Survival-Prediction`）
- Personal Access Token 或 SSH密钥

## 🚀 手动上传步骤

### 步骤1：在GitHub网站创建新仓库

1. 登录 [GitHub.com](https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `CKD5-CAP-Survival-Prediction`
   - **Description**: `CKD5期合并CAP生存预测系统 - 基于机器学习的Web应用`
   - **Visibility**: 选择 Public 或 Private
   - **不要**勾选 "Add a README file"（因为我们已经有了）
   - **不要**勾选 "Add .gitignore"（因为我们已经有了）
   - **不要**选择 License（因为我们已经有了）
4. 点击 "Create repository"

### 步骤2：配置认证（选择其中一种方式）

#### 方式A：Personal Access Token（推荐）

1. 在GitHub中生成Token：
   - 进入 Settings → Developer settings → Personal access tokens → Tokens (classic)
   - 点击 "Generate new token (classic)"
   - 设置过期时间（建议90天或更长）
   - 选择权限：勾选 `repo`（完整仓库访问权限）
   - 点击 "Generate token"
   - **重要**：复制并保存token（只显示一次）

2. 配置Git使用Token：
```bash
# 设置Git记住凭据（可选）
git config --global credential.helper store
```

#### 方式B：SSH密钥（如果已配置）

如果您已经配置了SSH密钥，可以直接使用SSH URL。

### 步骤3：添加远程仓库并推送

在项目目录中打开终端，执行以下命令：

#### 使用HTTPS（Personal Access Token）

```bash
# 添加远程仓库（替换YOUR_USERNAME为您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/CKD5-CAP-Survival-Prediction.git

# 推送代码到GitHub
git push -u origin master
```

**注意**：推送时会要求输入用户名和密码：
- 用户名：您的GitHub用户名
- 密码：使用您生成的Personal Access Token（不是GitHub密码）

#### 使用SSH（如果已配置SSH密钥）

```bash
# 添加远程仓库（替换YOUR_USERNAME为您的GitHub用户名）
git remote add origin git@github.com:YOUR_USERNAME/CKD5-CAP-Survival-Prediction.git

# 推送代码到GitHub
git push -u origin master
```

### 步骤4：验证上传成功

1. 在浏览器中访问您的GitHub仓库页面
2. 确认所有文件都已上传
3. 检查README.md是否正确显示
4. 确认提交历史正确

## 🔧 可能遇到的问题及解决方案

### 问题1：推送被拒绝（rejected）

**原因**：远程仓库有本地没有的提交

**解决方案**：
```bash
# 先拉取远程更改
git pull origin master --allow-unrelated-histories

# 然后推送
git push origin master
```

### 问题2：认证失败

**HTTPS方式**：
- 确认用户名正确
- 确认使用的是Personal Access Token而不是GitHub密码
- 检查Token权限是否包含`repo`

**SSH方式**：
- 确认SSH密钥已添加到GitHub账户
- 测试SSH连接：`ssh -T git@github.com`

### 问题3：分支名称问题

如果您的默认分支是`main`而不是`master`：

```bash
# 重命名本地分支
git branch -M main

# 推送到main分支
git push -u origin main
```

## 📁 项目文件结构确认

上传后，您的GitHub仓库应该包含以下文件：

```
CKD5-CAP-Survival-Prediction/
├── .gitignore
├── README.md
├── app.py                    # 主应用文件
├── config.py                 # 配置文件
├── requirements.txt          # 依赖列表
├── sample_data.csv          # 示例数据
├── run.bat                  # Windows启动脚本
├── run.sh                   # Linux/Mac启动脚本
├── models/                  # 模型文件目录
│   ├── XGBoost_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── knn_imputer.pkl
└── templates/               # HTML模板目录
    ├── index.html
    ├── predict.html
    ├── batch_predict.html
    └── results.html
```

## 🎉 上传完成后的后续步骤

1. **更新README.md**：添加GitHub仓库的具体信息
2. **设置仓库描述**：在GitHub仓库页面添加描述和标签
3. **创建Release**：为项目创建版本发布
4. **添加License**：如果需要，添加开源许可证
5. **设置GitHub Pages**：如果想要在线演示（可选）

## 📞 需要帮助？

如果在上传过程中遇到任何问题，请提供具体的错误信息，我可以帮助您解决。

---

**当前项目路径**：`d:\研究生学习-刘宽\项目代码\慢性肾病肺癌愈合\GitHub项目`

**Git状态**：已初始化，包含2个提交
- 初始提交：完整项目文件
- 安全修复提交：Flask配置和依赖版本修复