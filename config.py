#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 特征名称映射
FEATURE_NAME_MAP = {
    'CURB65评分': {'chinese': 'CURB65评分', 'english': 'CURB65 Score'},
    '重度肺炎': {'chinese': '重度肺炎', 'english': 'Severe Pneumonia'},
    'DDimer': {'chinese': 'D-Dimer', 'english': 'D-Dimer'},
    '炎症累及肺炎数≥3': {'chinese': '炎症累及肺炎数≥3', 'english': 'Inflammation ≥3 Lobes'},
    'PLR': {'chinese': '血小板淋巴细胞比值', 'english': 'Platelet-to-Lymphocyte Ratio'},
    '年龄': {'chinese': '年龄', 'english': 'Age'},
    '炎症累计双肺': {'chinese': '炎症累计双肺', 'english': 'Bilateral Lung Inflammation'},
    '冠心病': {'chinese': '冠心病', 'english': 'Coronary Heart Disease'},
    '饮酒史': {'chinese': '饮酒史', 'english': 'Drinking History'},
    'UA': {'chinese': '尿酸', 'english': 'Uric Acid'}
}