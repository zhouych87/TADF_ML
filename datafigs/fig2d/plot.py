# -*- coding: utf-8 -*-
"""
使用已有的预测数据绘制散点图 - 支持选择max_features
输入文件: prediction_data_*.csv
字体大小与opt2.py保持一致
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========== 配置参数（请在此修改） ==========
input_file = 'prediction_data_morgan_512m3_xgb.csv'  # 预测数据文件
output_prefix = 'scatter_plot_max'

# 选择要绘制的max_features值
# 设置为 None 则绘制所有max_features
# 设置为具体数值则只绘制该值，例如: selected_max_features = [6, 10, 20]
selected_max_features = 69  # 可选: None, 6, 10, 20, 50, 100, [6, 10, 20]

# ========== 字体大小设置（与opt2.py保持一致） ==========
FONT_SIZE_LABEL = 14      # 坐标轴标签字体大小
FONT_SIZE_TITLE = 12      # 标题字体大小
FONT_SIZE_LEGEND = 15     # 图例字体大小
FONT_SIZE_TICK = 10       # 刻度标签字体大小（默认值）
# ===================================================

# 目标变量名称（用于坐标轴标签）
target_name = 'λem'   # 可改为 'ΔEst' 等
target_unit = 'nm'    # 可改为 'eV' 等

# 设置刻度标签字体大小
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK

# 读取数据
df = pd.read_csv(input_file)
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 获取所有max_features值
all_mf_values = sorted(df['max_features'].unique())
print(f"max_features 可选值: {all_mf_values}")

# 确定要绘制的max_features列表
if selected_max_features is None:
    mf_to_plot = all_mf_values
elif isinstance(selected_max_features, (int, float)):
    mf_to_plot = [selected_max_features]
elif isinstance(selected_max_features, list):
    mf_to_plot = selected_max_features
else:
    mf_to_plot = all_mf_values

print(f"将要绘制的 max_features 值: {mf_to_plot}")

# 验证选中的值是否存在
for mf in mf_to_plot:
    if mf not in all_mf_values:
        print(f"警告: max_features={mf} 不存在于数据中，已跳过")
        mf_to_plot = [m for m in mf_to_plot if m in all_mf_values]

if not mf_to_plot:
    print("错误: 没有有效的 max_features 值可绘制")
    exit()

# 获取每个max_features的最佳模型（按test_r2最高）
best_per_mf = {}

for mf in mf_to_plot:
    mf_data = df[df['max_features'] == mf]
    
    # 计算每个random_state的测试集R²
    results = []
    for rs in mf_data['random_state'].unique():
        rs_data = mf_data[mf_data['random_state'] == rs]
        test_data = rs_data[rs_data['dataset'] == 'test']
        if len(test_data) > 0:
            test_r2 = r2_score(test_data['experimental'], test_data['predicted'])
            results.append({'random_state': rs, 'test_r2': test_r2})
    
    if results:
        best_result = max(results, key=lambda x: x['test_r2'])
        best_rs = best_result['random_state']
        best_test_r2 = best_result['test_r2']
        
        # 获取该random_state的完整数据
        best_data = mf_data[mf_data['random_state'] == best_rs]
        train_data = best_data[best_data['dataset'] == 'train']
        test_data = best_data[best_data['dataset'] == 'test']
        
        best_per_mf[mf] = {
            'random_state': best_rs,
            'test_r2': best_test_r2,
            'y_train': train_data['experimental'].values,
            'y_train_pred': train_data['predicted'].values,
            'y_test': test_data['experimental'].values,
            'y_test_pred': test_data['predicted'].values
        }
        
        print(f"max_features={mf}: best random_state={best_rs}, test_r2={best_test_r2:.4f}")

# 绘制每个max_features的散点图
for mf, best in best_per_mf.items():
    print(f"\n绘制 max_features={mf} 的散点图...")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制训练集：浅蓝色空心三角形
    ax.scatter(best['y_train'], best['y_train_pred'], alpha=0.6, s=50, 
               c='lightblue', marker='^', edgecolors='steelblue', linewidth=1, label='Training Set')
    
    # 绘制测试集：橘黄色空心圆圈
    ax.scatter(best['y_test'], best['y_test_pred'], alpha=0.6, s=50, 
               c='orange', marker='o', edgecolors='darkorange', linewidth=1, label='Test Set')
    
    # 计算训练集和测试集性能
    train_r2 = r2_score(best['y_train'], best['y_train_pred'])
    train_mae = mean_absolute_error(best['y_train'], best['y_train_pred'])
    train_rmse = np.sqrt(mean_squared_error(best['y_train'], best['y_train_pred']))
    
    test_r2 = best['test_r2']
    test_mae = mean_absolute_error(best['y_test'], best['y_test_pred'])
    test_rmse = np.sqrt(mean_squared_error(best['y_test'], best['y_test_pred']))
    
    # 坐标轴范围设置
    all_y = np.concatenate([best['y_train'], best['y_test']])
    all_pred = np.concatenate([best['y_train_pred'], best['y_test_pred']])
    
    data_min = min(all_y.min(), all_pred.min())
    data_max = max(all_y.max(), all_pred.max())
    data_range = data_max - data_min
    
    margin = max(data_range * 0.05, 0.005)
    x_min = data_min - margin
    x_max = data_max + margin
    y_min = data_min - margin
    y_max = data_max + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.plot([x_min, x_max], [x_min, x_max], 'k-', linewidth=2, label='Perfect Prediction')
    
    # ========== 字体设置（与opt2.py一致） ==========
    # 坐标轴标签：fontsize=14
    ax.set_xlabel(f'Experimental {target_name} ({target_unit})', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(f'Predicted {target_name} ({target_unit})', fontsize=FONT_SIZE_LABEL)
    
    # 标题：fontsize=12
    ax.set_title(f'max_features={mf}, best random_state={best["random_state"]}\nRF Model Performance for {target_name} Prediction', 
                 fontsize=FONT_SIZE_TITLE)
    # =============================================
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 创建图例（包含性能指标）
    train_legend = mlines.Line2D([], [], color='lightblue', marker='^', linestyle='None',
                                  markersize=10, label=f'Training: R²={train_r2:.3f}, RMSE={train_rmse:.2f}')
    test_legend = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                                 markersize=10, label=f'Test: R²={test_r2:.3f}, RMSE={test_rmse:.2f}')
    perfect_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=2, label='Perfect Prediction')
    
    # 图例：fontsize=10
    ax.legend(handles=[train_legend, test_legend, perfect_line], 
              loc='lower right', 
              fontsize=FONT_SIZE_LEGEND)
    
    # 设置等比例坐标轴
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = f'{output_prefix}_mf{mf}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  散点图已保存到: {output_file}")

print("\n所有散点图绘制完成！")
print(f"字体设置: 标签={FONT_SIZE_LABEL}, 标题={FONT_SIZE_TITLE}, 图例={FONT_SIZE_LEGEND}, 刻度={FONT_SIZE_TICK}")
print(f"绘制的 max_features 值: {mf_to_plot}")