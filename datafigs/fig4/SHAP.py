# -*- coding: utf-8 -*-
"""
SHAP分析 - 蜂群图绘制
对训练好的最佳模型进行SHAP分析
"""

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ========== 配置参数 ==========
# 选择要分析的描述符和模型
desc_name = "rdkit"      # 可选: rdkit, opt3d, morgan
model_type = "xgb"       # 可选: xgb, rf

# 文件路径配置
if desc_name == "rdkit":
    if model_type == "xgb":
        model_file = "best_model_rdkit_xgb.pkl"
        feature_file = "best_model_rdkit_xgb_features.pkl"
        data_file = "em_rdkit_cleaned_processed_sorted.pkl"
        original_csv_file = "em_rdkit_cleaned_processed.csv"  # 原始CSV文件
        output_prefix = "shap_rdkit_xgb"
    else:
        model_file = "best_model_rdkit_rf.pkl"
        feature_file = "best_model_rdkit_rf_features.pkl"
        data_file = "Est_rdkit_cleaned_sorted.pkl"
        original_csv_file = "Est_rdkit_cleaned_sorted.csv"
        output_prefix = "shap_rdkit_rf"
elif desc_name == "opt3d":
    if model_type == "xgb":
        model_file = "best_model_opt3d_xgb.pkl"
        feature_file = "best_model_opt3d_xgb_features.pkl"
        data_file = "opt3d_sl0.1_nsc60_xgb_sorted.pkl"
        original_csv_file = "Est_opt3d_0.1_cleaned_sorted.csv"
        output_prefix = "shap_opt3d_xgb"
    else:
        model_file = "best_model_opt3d_rf.pkl"
        feature_file = "best_model_opt3d_rf_features.pkl"
        data_file = "opt3d_sl0.02_nsc380_rf_sorted.pkl"
        original_csv_file = "Est_opt3d_0.02_cleaned_sorted.csv"
        output_prefix = "shap_opt3d_rf"
elif desc_name == "morgan":
    if model_type == "xgb":
        model_file = "best_model_morgan_xgb.pkl"
        feature_file = "best_model_morgan_xgb_features.pkl"
        data_file = "Est_1024m1_cleaned_sorted.pkl"
        original_csv_file = "Est_1024m1_cleaned.csv"
        output_prefix = "shap_morgan_xgb"
    else:
        model_file = "best_model_morgan_rf.pkl"
        feature_file = "best_model_morgan_rf_features.pkl"
        data_file = "Est_1024m1_cleaned_sorted.pkl"
        original_csv_file = "Est_1024m1_cleaned.csv"
        output_prefix = "shap_morgan_rf"
else:
    raise ValueError("desc_name must be 'rdkit', 'opt3d', or 'morgan'")

print("="*60)
print(f"SHAP分析: {desc_name} - {model_type.upper()}")
print("="*60)

# ========== 0. 从原始CSV获取真实特征名称 ==========
print("\n【第零步】从原始CSV文件获取真实特征名称...")

# 读取原始CSV文件获取列名
original_df = pd.read_csv(original_csv_file, encoding='utf-8-sig')
print(f"原始CSV列数: {len(original_df.columns)}")
print(f"原始CSV列名（前10个）: {original_df.columns[:10].tolist()}")

# 排除最后一列（目标变量）
if 'Est' in original_df.columns:
    original_feature_names = original_df.columns[:-1].tolist()
elif 'label' in original_df.columns:
    original_feature_names = original_df.columns[:-1].tolist()
else:
    # 如果最后一列不是Est或label，假设最后一列是目标变量
    original_feature_names = original_df.columns[:-1].tolist()

print(f"真实特征名称数量: {len(original_feature_names)}")
print(f"真实特征名称（前10个）: {original_feature_names[:10]}")

# ========== 1. 加载模型和选中的特征 ==========
print("\n【第一步】加载模型和选中的特征...")
with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(feature_file, 'rb') as f:
    selected_features = pickle.load(f)

print(f"模型加载成功")
print(f"选中特征数量: {len(selected_features)}")
print(f"选中的特征（前5个）: {selected_features[:5]}")

# ========== 2. 加载原始数据 ==========
print("\n【第二步】加载原始数据...")
with open(data_file, 'rb') as f:
    data = pickle.load(f)

if isinstance(data, dict):
    X_full = data['X']
    y = data['y']
    all_feature_names = data['feature_names']
    print(f"数据中的特征名（前5个）: {all_feature_names[:5]}")
else:
    if 'expt' in data.columns:
        X_full = data.drop(columns=['expt']).values
        y = data['expt'].values
        all_feature_names = data.columns[:-1].tolist()
    else:
        X_full = data.values[:, :-1]
        y = data.values[:, -1]
        all_feature_names = [f'feature_{i}' for i in range(X_full.shape[1])]

print(f"原始数据形状: X={X_full.shape}, y={y.shape}")

# ========== 3. 核心：建立特征索引映射 ==========
print("\n【第三步】建立特征索引映射...")

# 方法：判断selected_features中是数字索引还是特征名
if len(selected_features) > 0:
    # 检查selected_features的第一个元素是否是数字（字符串形式的数字）
    first_feat = str(selected_features[0])
    if first_feat.isdigit() or (first_feat.startswith('feature_') and first_feat.split('_')[1].isdigit()):
        # selected_features是索引，需要映射到真实名称
        print("检测到selected_features是索引格式，正在映射到真实特征名称...")
        
        # 提取索引数字
        if first_feat.startswith('feature_'):
            selected_indices = [int(f.split('_')[1]) for f in selected_features]
        else:
            selected_indices = [int(f) for f in selected_features]
        
        # 根据索引获取真实特征名称
        real_feature_names = [original_feature_names[idx] for idx in selected_indices]
        
        print(f"索引范围: {min(selected_indices)} - {max(selected_indices)}")
    else:
        # selected_features已经是特征名，直接使用
        print("selected_features已经是特征名格式")
        # 创建特征名到真实名称的映射（处理可能的不匹配）
        name_mapping = {}
        for i, orig_name in enumerate(original_feature_names):
            name_mapping[orig_name] = orig_name
            # 也尝试匹配可能的简化名称
            name_mapping[f'feature_{i}'] = orig_name
        
        real_feature_names = []
        for feat in selected_features:
            if feat in name_mapping:
                real_feature_names.append(name_mapping[feat])
            elif feat.startswith('feature_') and feat in name_mapping:
                real_feature_names.append(name_mapping[feat])
            else:
                # 尝试按位置匹配
                try:
                    idx = int(feat.split('_')[-1]) if feat.startswith('feature_') else int(feat)
                    if idx < len(original_feature_names):
                        real_feature_names.append(original_feature_names[idx])
                    else:
                        real_feature_names.append(feat)
                except:
                    real_feature_names.append(feat)

print(f"映射后的真实特征名称数量: {len(real_feature_names)}")
print(f"映射后的真实特征名称（前10个）: {real_feature_names[:10]}")

# ========== 4. 获取选中特征对应的数据索引 ==========
print("\n【第四步】获取选中特征的数据索引...")

# 创建原始特征名到索引的映射
orig_name_to_idx = {name: i for i, name in enumerate(original_feature_names)}

# 获取真实特征名称在原始数据中的索引
selected_idx = []
for real_name in real_feature_names:
    if real_name in orig_name_to_idx:
        selected_idx.append(orig_name_to_idx[real_name])
    else:
        # 尝试在all_feature_names中查找
        if real_name in feature_to_idx:
            selected_idx.append(feature_to_idx[real_name])
        else:
            print(f"警告: 特征 '{real_name}' 未找到")

print(f"成功匹配 {len(selected_idx)} 个特征")

# 选择特征
X_selected = X_full[:, selected_idx]

print(f"选中数据形状: X={X_selected.shape}")

# ========== 5. 数据预处理 ==========
print("\n【第五步】数据预处理...")
float32_max = np.finfo(np.float32).max
X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
X_selected = np.clip(X_selected, -float32_max, float32_max)
X_selected = X_selected.astype(np.float64)

# ========== 6. 创建SHAP解释器 ==========
print("\n【第六步】创建SHAP解释器...")

# 使用全部样本
n_samples = X_selected.shape[0]
X_sample = X_selected  # 使用全部数据
y_sample = y           # 使用全部标签

print(f"使用全部 {n_samples} 个样本进行SHAP分析")

print("创建TreeExplainer...")
explainer = shap.TreeExplainer(model)

print("计算SHAP值（可能需要几分钟）...")
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

print(f"SHAP值形状: {shap_values.shape}")

# ========== 7. 绘制蜂群图 ==========
print("\n【第七步】绘制蜂群图...")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

top_n = min(10, len(real_feature_names))

feature_importance = np.abs(shap_values).mean(0)
top_indices = np.argsort(feature_importance)[-top_n:][::-1]

selected_features_top = [real_feature_names[i] for i in top_indices]
shap_values_top = shap_values[:, top_indices]
X_sample_top = X_sample[:, top_indices]

selected_features_simple = [f[:35] + '...' if len(f) > 35 else f for f in selected_features_top]

plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values_top, X_sample_top, 
                  feature_names=selected_features_simple, 
                  show=False, max_display=top_n)
plt.tight_layout()
plt.savefig(f'{output_prefix}_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"蜂群图已保存: {output_prefix}_beeswarm.png")

# ========== 8. 绘制条形图 ==========
print("\n【第八步】绘制特征重要性条形图...")

mean_shap = np.abs(shap_values).mean(0)
shap_importance = pd.DataFrame({
    'feature': real_feature_names,
    'mean_shap': mean_shap
})
shap_importance = shap_importance.sort_values('mean_shap', ascending=False).head(top_n)

shap_importance['feature_simple'] = shap_importance['feature'].apply(
    lambda x: x[:35] + '...' if len(x) > 35 else x
)

plt.figure(figsize=(10, 8))
plt.barh(shap_importance['feature_simple'], shap_importance['mean_shap'], color='steelblue')
plt.xlabel('Mean |SHAP value|', fontsize=20)
plt.ylabel('Feature', fontsize=20)
plt.title(f'Feature Importance (SHAP) - {desc_name.upper()} {model_type.upper()}', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_prefix}_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"条形图已保存: {output_prefix}_bar.png")

# ========== 9. 保存SHAP值（使用真实特征名） ==========
print("\n【第九步】保存SHAP值...")

shap_df = pd.DataFrame(shap_values, columns=real_feature_names)
shap_df.to_csv(f'{output_prefix}_shap_values.csv', index=False)
print(f"SHAP值已保存: {output_prefix}_shap_values.csv")

sample_df = pd.DataFrame(X_sample, columns=real_feature_names)
sample_df['target'] = y_sample
sample_df.to_csv(f'{output_prefix}_sample_data.csv', index=False)
print(f"样本数据已保存: {output_prefix}_sample_data.csv")

# ========== 10. 保存特征重要性 ==========
shap_importance_full = pd.DataFrame({
    'feature': real_feature_names,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=False)
shap_importance_full.to_csv(f'{output_prefix}_feature_importance.csv', index=False)
print(f"特征重要性已保存: {output_prefix}_feature_importance.csv")

# ========== 11. 保存特征映射文件 ==========
mapping_df = pd.DataFrame({
    'index': range(len(real_feature_names)),
    'real_feature_name': real_feature_names,
    'original_selected_name': selected_features if len(selected_features) == len(real_feature_names) else ['N/A']*len(real_feature_names)
})
mapping_df.to_csv(f'{output_prefix}_feature_mapping.csv', index=False)
print(f"特征映射文件已保存: {output_prefix}_feature_mapping.csv")

print("\n" + "="*60)
print("SHAP分析完成！")
print("="*60)
print(f"\n输出文件:")
print(f"  1. {output_prefix}_beeswarm.png - 蜂群图")
print(f"  2. {output_prefix}_bar.png - 条形图")
print(f"  3. {output_prefix}_shap_values.csv - SHAP值矩阵")
print(f"  4. {output_prefix}_sample_data.csv - 样本数据")
print(f"  5. {output_prefix}_feature_importance.csv - 特征重要性排序")
print(f"  6. {output_prefix}_feature_mapping.csv - 特征映射文件")
print(f"\n特征名称示例: {real_feature_names[:5]}")