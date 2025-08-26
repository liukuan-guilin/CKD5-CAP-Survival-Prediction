#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CKD5期合并CAP生存预测Web应用
基于Flask框架的用户友好预测系统
支持实时SHAP可解释性分析

作者: 开发团队
版本: 1.0
日期: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file, session
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib
import os
import shap
import io
import base64
from datetime import datetime
import warnings
import glob
import secrets
from config import FEATURE_NAME_MAP
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
# 使用环境变量或随机生成的安全密钥
app.secret_key = os.getenv('FLASK_SECRET_KEY') or secrets.token_hex(32)

class CKD5CAPSurvivalPredictionApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None  # 新增imputer
        self.feature_columns = []
        self.shap_analyzer = None
        self.feature_descriptions = {}
        self.chinese_feature_names = {}
        self.english_feature_names = {}
        self.translations = {}
        
        # 定义特征列表和描述
        self.init_feature_definitions()
        
        # 初始化多语言支持
        self.init_translations()
        
        # 自动查找并加载最新模型
        self.load_latest_model()
    
    def init_feature_definitions(self):
        """初始化特征定义和描述"""
        # 从配置文件加载特征映射
        self.feature_columns = list(FEATURE_NAME_MAP.keys())
        self.chinese_feature_names = {k: v['chinese'] for k, v in FEATURE_NAME_MAP.items()}
        self.english_feature_names = {k: v['english'] for k, v in FEATURE_NAME_MAP.items()}

        # 特征描述和正常范围（可以根据需要扩展）
        self.feature_descriptions = {
            'CURB65评分': {'desc': 'CURB65评分，用于评估肺炎严重程度', 'range': '0-5分', 'unit': '分'},
            '重度肺炎': {'desc': '是否为重度肺炎', 'range': '0或1', 'unit': ''},
            'DDimer': {'desc': 'D-二聚体，反映凝血功能', 'range': '<0.5', 'unit': 'mg/L'},
            '炎症累及肺炎数≥3': {'desc': '肺部炎症是否累及3个或以上肺叶', 'range': '0或1', 'unit': ''},
            'PLR': {'desc': '血小板淋巴细胞比值', 'range': '100-300', 'unit': ''},
            '年龄': {'desc': '患者年龄', 'range': '18-100', 'unit': '岁'},
            '炎症累计双肺': {'desc': '是否为双肺炎症', 'range': '0或1', 'unit': ''},
            '冠心病': {'desc': '是否患有冠心病', 'range': '0或1', 'unit': ''},
            '饮酒史': {'desc': '是否有饮酒史', 'range': '0或1', 'unit': ''},
            'UA': {'desc': '尿酸', 'range': '150-420', 'unit': 'μmol/L'}
        }
    
    def init_translations(self):
        """初始化多语言翻译"""
        self.translations = {
            'zh': {
                'title': 'CKD5期合并CAP生存情况预测',
                'subtitle': '基于机器学习的生存预测与分析',
                'single_prediction': '单患者预测',
                'batch_prediction': '批量预测',
                'basic_info': '基本信息',
                'medical_history': '既往病史',
                'vital_signs': '生命体征',
                'imaging': '影像学检查',
                'lab_tests': '实验室检查',
                'blood_routine': '血常规',
                'biochemistry': '生化检查',
                'coagulation': '凝血功能',
                'predict_button': '开始预测',
                'clear_button': '清空表单',
                'prediction_result': '预测结果',
                'survival_probability': '生存概率',
                'death_probability': '死亡概率',
                'risk_level': '风险等级',
                'high_risk': '高风险',
                'medium_risk': '中风险',
                'low_risk': '低风险',
                'feature_importance': '特征重要性',
                'shap_analysis': 'SHAP可解释性分析',
                'upload_file': '上传文件',
                'download_results': '下载结果',
                'model_info': '模型信息',
                'language': '语言',
                'chinese': '中文',
                'english': 'English',
                'loading': '预测中，请稍候...',
                'error': '错误',
                'success': '成功',
                'male': '男',
                'female': '女',
                'yes': '是',
                'no': '否',
                'select': '请选择'
            },
            'en': {
                'title': 'CKD Stage 5 with CAP Survival Prediction',
                'subtitle': 'Machine Learning-Based Survival Prediction and Analysis',
                'single_prediction': 'Single Patient Prediction',
                'batch_prediction': 'Batch Prediction',
                'basic_info': 'Basic Information',
                'medical_history': 'Medical History',
                'vital_signs': 'Vital Signs',
                'imaging': 'Imaging Studies',
                'lab_tests': 'Laboratory Tests',
                'blood_routine': 'Blood Routine',
                'biochemistry': 'Biochemistry',
                'coagulation': 'Coagulation Function',
                'predict_button': 'Start Prediction',
                'clear_button': 'Clear Form',
                'prediction_result': 'Prediction Result',
                'survival_probability': 'Survival Probability',
                'death_probability': 'Death Probability',
                'risk_level': 'Risk Level',
                'high_risk': 'High Risk',
                'medium_risk': 'Medium Risk',
                'low_risk': 'Low Risk',
                'feature_importance': 'Feature Importance',
                'shap_analysis': 'SHAP特征分析',
                'upload_file': 'Upload File',
                'download_results': 'Download Results',
                'model_info': 'Model Information',
                'language': 'Language',
                'chinese': '中文',
                'english': 'English',
                'loading': 'Predicting, please wait...',
                'error': 'Error',
                'success': 'Success',
                'male': 'Male',
                'female': 'Female',
                'yes': 'Yes',
                'no': 'No',
                'select': 'Please Select'
            }
        }
    
    def find_latest_model_directory(self):
        """查找最新的模型目录"""
        pattern = 'survival_prediction_results_*'
        directories = glob.glob(pattern)
        
        if not directories:
            return None
        
        # 按时间戳排序，获取最新的
        directories.sort(reverse=True)
        
        # 检查目录中是否包含必要文件
        for directory in directories:
            model_files = glob.glob(os.path.join(directory, 'best_model_*.pkl'))
            scaler_file = os.path.join(directory, 'scaler.pkl')
            
            if model_files and os.path.exists(scaler_file):
                return directory
        
        return None
    
    def load_specific_model(self, model_path):
        """加载指定的模型文件"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # 加载指定模型
            self.model = joblib.load(model_path)
            
            # 检查模型类型和特征数量
            if hasattr(self.model, 'estimators_'):
                # 获取第一个基模型来检查特征数量
                base_model = self.model.estimators_[0][1] if hasattr(self.model.estimators_[0], '__len__') else self.model.estimators_[0]
                if hasattr(base_model, 'n_features_in_'):
                    expected_features = base_model.n_features_in_
                    
                    if expected_features != len(self.feature_columns):
                        # 尝试从模型目录加载特征信息
                        model_dir = os.path.dirname(model_path)
                        self.load_feature_info_from_model_dir(model_dir)
            
            # 尝试加载对应的标准化器
            model_dir = os.path.dirname(model_path)
            # 优先加载10个特征的标准化器
            scaler_10_path = os.path.join(model_dir, 'scaler_10_features.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            
            if os.path.exists(scaler_10_path):
                self.scaler = joblib.load(scaler_10_path)
            elif os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                pass

            # 关键修复：从加载的scaler中获取正确的特征顺序
            if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
                self.feature_columns = self.scaler.feature_names_in_.tolist()

            # 尝试加载对应的imputer
            imputer_path = os.path.join(model_dir, 'knn_imputer.pkl')
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
            else:
                self.imputer = None # 明确设置为None
            
            # 初始化SHAP解释器
            if self.model and len(self.feature_columns) > 0:
                try:
                    # 检查模型类型并选择合适的SHAP解释器
                    model_type = type(self.model).__name__
                    
                    # 创建示例数据
                    if hasattr(self, 'scaler') and self.scaler is not None:
                        # 使用标准化后的数据范围
                        sample_data = np.random.normal(0, 1, (50, len(self.feature_columns)))
                    else:
                        # 使用合理的数据范围
                        sample_data = np.random.randn(50, len(self.feature_columns))
                    
                    # 根据模型类型选择解释器
                    if 'VotingClassifier' in model_type:
                        # VotingClassifier使用Kernel分析器
                        self.shap_analyzer = shap.KernelExplainer(self.model.predict_proba, sample_data)
                    elif hasattr(self.model, 'estimators_') and ('XGB' in model_type or 'RandomForest' in model_type):
                        # XGBoost和RandomForest使用Tree分析器
                        self.shap_analyzer = shap.TreeExplainer(self.model)
                    elif hasattr(self.model, 'predict_proba'):  # 支持概率预测的模型
                        # 尝试使用Tree分析器，失败则使用Kernel分析器
                        try:
                            self.shap_analyzer = shap.TreeExplainer(self.model)
                        except Exception as tree_error:
                            self.shap_analyzer = shap.KernelExplainer(self.model.predict_proba, sample_data)
                    else:
                        self.shap_analyzer = None
                except Exception as e:
                    self.shap_analyzer = None
            else:
                self.shap_analyzer = None
            
            return True
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def load_feature_info_from_model_dir(self, model_dir):
        """从模型目录加载特征信息"""
        try:
            # 优先尝试加载feature_names.pkl文件
            feature_names_file = os.path.join(model_dir, 'feature_names.pkl')
            if os.path.exists(feature_names_file):
                import pickle
                with open(feature_names_file, 'rb') as f:
                    model_features = pickle.load(f)
                # 更新特征列表
                self.feature_columns = model_features
                return True
            
            # 如果没有feature_names.pkl，尝试从预处理摘要文件解析
            summary_file = os.path.join(model_dir, 'preprocessing_summary.csv')
            if os.path.exists(summary_file):
                summary_df = pd.read_csv(summary_file)
                
                # 查找LASSO选择的特征列表
                lasso_features_row = summary_df[summary_df['项目'] == 'LASSO选择的特征列表']
                if not lasso_features_row.empty:
                    features_str = lasso_features_row['数值/内容'].iloc[0]
                    # 解析特征列表字符串
                    import ast
                    model_features = ast.literal_eval(features_str)
                    # 更新特征列表
                    self.feature_columns = model_features
                    return True
                
                # 如果没有LASSO特征列表，尝试Feature列
                if 'Feature' in summary_df.columns:
                    model_features = summary_df['Feature'].tolist()
                    # 更新特征列表
                    self.feature_columns = model_features
                    return True
        except Exception as e:
            pass
        return False
    
    def load_latest_model(self):
        """加载指定的模型和预处理器"""
        model_path = os.path.join('models', 'XGBoost_model.pkl')
        scaler_path = os.path.join('models', 'scaler.pkl')

        # 加载指定的模型和scaler

        try:
            if not os.path.exists(model_path):
                return
            if not os.path.exists(scaler_path):
                return

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            # 初始化SHAP解释器
            if self.model and hasattr(self.model, 'predict_proba'):
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception as e:
                    self.explainer = None
            else:
                self.explainer = None

        except Exception as e:
            pass
    
    def preprocess_input_data(self, input_data):
        """预处理输入数据"""
        try:
            # 1. 将输入数据（字典）的中文键转换为英文键
            # 同时处理''和None为np.nan
            input_data_english_keys = {}
            for eng_key, names in FEATURE_NAME_MAP.items():
                chinese_key = names['chinese']
                value = input_data.get(chinese_key)
                if value is None or value == '':
                    input_data_english_keys[eng_key] = np.nan
                else:
                    input_data_english_keys[eng_key] = value
            
            # 2. 创建一个包含所有模型所需特征的DataFrame
            processed_df = pd.DataFrame([input_data_english_keys], columns=self.feature_columns)

            # 3. 缺失值填充
            missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
            if missing_cols:
                if self.imputer:
                    # 使用imputer填充，注意imputer输出的是numpy数组
                    imputed_values = self.imputer.transform(processed_df)
                    processed_df = pd.DataFrame(imputed_values, columns=self.feature_columns, index=processed_df.index)
                else:
                    # 定义默认值（使用英文键）
                    default_values = {
                        'CURB65Score': 1, 'SeverePneumonia': 0, 'DDimer': 0.3, 'MultipleLobeInvolvement': 0,
                        'PLR': 120, 'Age': 65, 'BilateralPneumonia': 0, 'CoronaryHeartDisease': 0,
                        'AlcoholHistory': 0, 'UA': 300
                    }
                    # 仅填充缺失的列
                    for col in missing_cols:
                        if col in default_values:
                            processed_df[col].fillna(default_values[col], inplace=True)

            # 4. 确保所有列都是数值类型
            for col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)

            # 6. 确保DataFrame的列顺序与模型训练时一致
            processed_df = processed_df[self.feature_columns]

            # 7. 检查并调整特征顺序以匹配scaler训练时的顺序
            if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
                scaler_features = list(self.scaler.feature_names_in_)
                current_features = list(processed_df.columns)
                
                # 如果特征顺序不一致，重新排序
                if scaler_features != current_features:
                    # 确保所有scaler期望的特征都存在
                    missing_features = set(scaler_features) - set(current_features)
                    if missing_features:
                        return None
                    
                    # 按照scaler期望的顺序重新排序
                    processed_df = processed_df[scaler_features]

            # 8. 数据标准化
            if self.scaler:
                scaled_data = self.scaler.transform(processed_df)
                # 将scaled_data转回DataFrame以保留列名，方便SHAP解释
                scaled_df = pd.DataFrame(scaled_data, columns=self.feature_columns)
                return scaled_df
            else:
                return processed_df

        except Exception as e:
            return None
    
    def predict_and_analyze(self, input_data):
        """进行预测并生成SHAP解释"""
        try:
            # 预处理数据
            X_processed = self.preprocess_input_data(input_data)
            if X_processed is None:
                # 预处理失败
                return None, None, None, None
            
            # 检查并调整特征顺序以匹配模型期望的顺序
            if hasattr(self.model, 'feature_names_in_'):
                model_feature_names = self.model.feature_names_in_
                
                # 检查特征名称是否匹配
                if not all(feature in X_processed.columns for feature in model_feature_names):
                    missing_features = [f for f in model_feature_names if f not in X_processed.columns]
                    return None, None, None, None
                
                # 按照模型期望的顺序重新排列特征
                if list(X_processed.columns) != list(model_feature_names):
                    X_processed = X_processed[model_feature_names]
            
            # 预测
            prediction = self.model.predict(X_processed)[0]
            prediction_proba = self.model.predict_proba(X_processed)[0]
            
            # 检查并处理NaN值
            if np.isnan(prediction_proba).any():
                prediction_proba = np.array([0.5, 0.5])  # 默认50%概率
                prediction = 0  # 默认预测为生存
            
            # SHAP解释
            feature_importance = None
            shap_values_obj = None
            if self.explainer is not None:
                try:
                    shap_values_obj = self.explainer(X_processed)
                    
                    # 根据SHAP返回的对象类型提取shap值数组
                    shap_vals_raw = shap_values_obj.values
                    
                    # 处理SHAP值的维度问题
                    if len(shap_vals_raw.shape) == 3:  # (n_samples, n_features, n_classes)
                        # 对于二分类，通常取正类（索引1）的SHAP值
                        shap_vals = shap_vals_raw[0, :, 1]
                    elif len(shap_vals_raw.shape) == 2:  # (n_samples, n_features)
                        shap_vals = shap_vals_raw[0]
                    else:
                        shap_vals = shap_vals_raw # 假设已经是正确的1D数组
                    
                    # 确保是1维数组
                    shap_vals = np.ravel(shap_vals)
                    original_vals = np.ravel(X_processed.iloc[0].values)
                    
                    # 获取特征重要性排名
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_columns,
                        'chinese_name': [self.chinese_feature_names.get(f, f) for f in self.feature_columns],
                        'english_name': [self.english_feature_names.get(f, f) for f in self.feature_columns],
                        'importance': np.abs(shap_vals),
                        'shap_value': shap_vals,
                        'original_value': original_vals
                    }).sort_values('importance', ascending=False)
                    
                except Exception as e:
                    feature_importance = None
                    shap_values_obj = None
            
            return prediction, prediction_proba, feature_importance, shap_values_obj
            
        except Exception as e:
            return None, None, None, None
    
    def generate_shap_plot(self, shap_values, feature_names):
        """生成SHAP图表"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 创建SHAP瀑布图
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title('SHAP Feature Importance Waterfall Plot', fontsize=14, fontweight='bold')
            
            # 保存为base64字符串
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
            
        except Exception as e:
            return None
    
    def generate_feature_importance_plot(self, feature_importance):
        """生成特征重要性柱状图"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 取前15个重要特征
            top_features = feature_importance.head(15)
            
            # 创建柱状图
            colors = ['red' if x < 0 else 'blue' for x in top_features['shap_value']]
            bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors, alpha=0.7)
            
            # 设置标签
            plt.yticks(range(len(top_features)), top_features['english_name'])
            plt.xlabel('SHAP Importance Value', fontsize=12)
            plt.title('Feature Importance Ranking (Top 15)', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
                plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # 保存为base64字符串
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
            
        except Exception as e:
            return None

# 初始化应用
predictor = CKD5CAPSurvivalPredictionApp()

@app.route('/')
def index():
    """主页"""
    # 获取当前语言，默认为中文
    lang = session.get('language', 'zh')
    
    return render_template('ckd5_cap_prediction.html', 
                         features=predictor.feature_columns,
                         feature_descriptions=predictor.feature_descriptions,
                         chinese_names=predictor.chinese_feature_names,
                         english_names=predictor.english_feature_names,
                         translations=predictor.translations[lang],
                         current_lang=lang)

@app.route('/set_language/<language>')
def set_language(language):
    """设置语言"""
    if language in ['zh', 'en']:
        session['language'] = language
    return jsonify({'status': 'success', 'language': session.get('language', 'zh')})

@app.route('/get_translations/<language>')
def get_translations(language):
    """获取翻译文本"""
    if language in predictor.translations:
        return jsonify(predictor.translations[language])
    return jsonify(predictor.translations['zh'])  # 默认返回中文

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 处理不同类型的请求数据
        if request.is_json:
            data = request.get_json()
        elif request.form:
            data = request.form.to_dict()
            # 转换数值类型
            for key, value in data.items():
                if value and value.replace('.', '').replace('-', '').isdigit():
                    data[key] = float(value)
        else:
            return jsonify({'error': '不支持的请求格式，请使用JSON或表单数据'}), 400
            
        if not data:
            return jsonify({'error': '无效的输入数据'}), 400
        
        # 进行预测和解释
        prediction, prediction_proba, feature_importance, shap_values_obj = predictor.predict_and_analyze(data)
        
        if prediction is None:
            return jsonify({'error': '预测失败，请检查输入数据和服务器日志'}), 400
        
        # 生成SHAP图表
        shap_plot = None
        importance_plot = None
        if shap_values_obj is not None and feature_importance is not None:
            try:
                shap_plot = predictor.generate_shap_plot(shap_values_obj, predictor.feature_columns)
                importance_plot = predictor.generate_feature_importance_plot(feature_importance)
            except Exception as e:
                pass
        
        # 检查并处理NaN值
        survival_prob = prediction_proba[0] if not np.isnan(prediction_proba[0]) else 0.5
        death_prob = prediction_proba[1] if not np.isnan(prediction_proba[1]) else 0.5
        
        # 生成SHAP解释HTML
        shap_explanation_html = None
        if feature_importance is not None and len(feature_importance) > 0:
            # 生成特征重要性的HTML表格
            top_features = feature_importance.head(5)
            shap_explanation_html = '<div class="shap-analysis">'
            shap_explanation_html += '<h6>主要影响因素（前5个）：</h6>'
            shap_explanation_html += '<div class="table-responsive">'
            shap_explanation_html += '<table class="table table-sm table-striped">'
            shap_explanation_html += '<thead><tr><th>特征</th><th>当前值</th><th>影响程度</th><th>影响方向</th></tr></thead>'
            shap_explanation_html += '<tbody>'
            
            for _, row in top_features.iterrows():
                impact_direction = '增加风险' if row['shap_value'] > 0 else '降低风险'
                impact_color = 'text-danger' if row['shap_value'] > 0 else 'text-success'
                shap_explanation_html += f'<tr>'
                shap_explanation_html += f'<td>{row["chinese_name"]}</td>'
                shap_explanation_html += f'<td>{row["original_value"]:.2f}</td>'
                shap_explanation_html += f'<td>{abs(row["shap_value"]):.3f}</td>'
                shap_explanation_html += f'<td class="{impact_color}">{impact_direction}</td>'
                shap_explanation_html += f'</tr>'
            
            shap_explanation_html += '</tbody></table></div>'
            
            # 添加SHAP图表（如果有）
            if shap_plot:
                shap_explanation_html += f'<div class="mt-3"><img src="data:image/png;base64,{shap_plot}" class="img-fluid" alt="SHAP瀑布图"></div>'
            if importance_plot:
                shap_explanation_html += f'<div class="mt-3"><img src="data:image/png;base64,{importance_plot}" class="img-fluid" alt="特征重要性图"></div>'
            
            shap_explanation_html += '</div>'
        
        # 准备返回结果
        result = {
            'prediction': int(prediction),
            'prediction_proba': {
                'survival': float(survival_prob),  # 生存概率
                'death': float(death_prob)         # 死亡概率
            },
            # 为前端兼容性添加直接字段
            'survival_probability': float(survival_prob),
            'death_probability': float(death_prob),
            'risk_level': '高风险' if death_prob > 0.7 else '中风险' if death_prob > 0.3 else '低风险',
            'feature_importance': feature_importance.head(10).to_dict('records') if feature_importance is not None else [],
            'shap_plot': shap_plot,
            'importance_plot': importance_plot,
            'shap_explanation': shap_explanation_html,  # 添加前端需要的字段
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'interpretation': {
                'survival_probability': f'{survival_prob*100:.1f}%',
                'death_probability': f'{death_prob*100:.1f}%',
                'top_risk_factors': feature_importance.head(5)[['english_name', 'shap_value']].to_dict('records') if feature_importance is not None else []
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'预测过程中发生严重错误，请联系管理员'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        # 检查是否有上传的文件
        if 'file' not in request.files:
            return jsonify({'error': '未找到上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 读取文件
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': '不支持的文件格式，请上传CSV或Excel文件'}), 400
        
        results = []
        
        # 逐行预测
        for index, row in df.iterrows():
            data = row.to_dict()
            prediction, prediction_proba, feature_importance, shap_values_obj = predictor.predict_and_analyze(data)
            
            if prediction is not None:
                results.append({
                    'row_index': index + 1,
                    'prediction': int(prediction),
                    'survival_probability': float(prediction_proba[0]),
                    'death_probability': float(prediction_proba[1]),
                    'risk_level': '高风险' if prediction_proba[1] > 0.7 else '中风险' if prediction_proba[1] > 0.3 else '低风险'
                })
            else:
                results.append({
                    'row_index': index + 1,
                    'error': '预测失败'
                })
        
        return jsonify({
            'total_rows': len(df),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'results': results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """模型信息接口"""
    try:
        info = {
            'model_loaded': predictor.model is not None,
            'scaler_loaded': predictor.scaler is not None,
            'analyzer_loaded': predictor.shap_analyzer is not None,
            'feature_count': len(predictor.feature_columns),
            'features': predictor.feature_columns,
            'feature_details': [
                {
                    'name': feature,
                    'chinese_name': predictor.chinese_feature_names.get(feature, feature),
                    'description': predictor.feature_descriptions.get(feature, {}).get('desc', ''),
                    'normal_range': predictor.feature_descriptions.get(feature, {}).get('range', ''),
                    'unit': predictor.feature_descriptions.get(feature, {}).get('unit', '')
                }
                for feature in predictor.feature_columns
            ]
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'获取模型信息失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 生产环境安全配置：仅监听本地接口
    app.run(debug=False, host='127.0.0.1', port=5000)