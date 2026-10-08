# -*- coding: utf-8 -*-
"""
Morgan指纹描述符超参数优化
使用特征重要性排序后的数据
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
import csv
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --------------------------
# 全局日志配置
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 已经预处理好的 DataFrame
file0 = Path('Est_1024m1_cleaned_sorted.pkl')  # 使用排序后的Morgan指纹数据
data_dict = pickle.load(open(file0, 'rb'))

# 从字典中提取数据
if isinstance(data_dict, dict):
    X_full = data_dict['X']
    y = data_dict['y']
    feature_names = data_dict['feature_names']
    processed_df = pd.DataFrame(X_full, columns=feature_names)
    processed_df['expt'] = y
else:
    processed_df = data_dict
    if isinstance(processed_df, np.ndarray):
        processed_df = pd.DataFrame(processed_df)
    processed_df = processed_df.rename(columns={processed_df.columns[-1]: 'expt'})

logger.info(f"数据形状: X={processed_df.drop(columns=['expt']).shape}, y={processed_df['expt'].shape}")
logger.info(f"特征数量: {len(processed_df.columns) - 1}")

# 2. 准备结果文件
result_csv = 'results_opt_morgan_1024m1.csv'
with open(result_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['max_features', 'random_state', 'best_cv_r2', 'test_r2', 'test_mae', 'test_rmse'])

# 3. 网格搜索参数
max_features_list = [20, 25, 30, 50, 100, 1026]
random_state_list = list(range(1, 11))

# 存储所有结果用于绘图和分析
all_results = []
all_prediction_data = []  # 存储所有预测数据
best_model_per_mf = {}  # 存储每个max_features的最佳模型信息

for mf in max_features_list:
    for rs in random_state_list:
        logger.info(f'=== Running max_features={mf}, random_state={rs} ===')

        def feature_selection(X, y,
                              corr_thresh=0.9,
                              max_rf_features=5,
                              random_state=42):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
            df = pd.DataFrame(X_scaled)
            corr = df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop_corr = [c for c in upper.columns if any(upper[c] > corr_thresh)]
        
            if len(to_drop_corr):
                y_series = pd.Series(y)
                keep_from_corr = []
                for col in to_drop_corr:
                    partners = upper.index[upper[col] > corr_thresh].tolist() + [col]
                    best = np.argmax([abs(df[c].corr(y_series)) for c in partners])
                    keep_from_corr.append(partners[best])
                keep_from_corr = list(set(keep_from_corr))
                final_drop = [c for c in to_drop_corr if c not in keep_from_corr]
            else:
                final_drop = []
        
            X_corr = np.delete(X_scaled, final_drop, axis=1)
            logger.info(f"After correlation pruning: {X_corr.shape}")
        
            reg = RandomForestRegressor(n_estimators=300,
                                        random_state=random_state,
                                        n_jobs=-1)
            reg.fit(X_corr, y)
        
            selector = SelectFromModel(estimator=reg,
                                       max_features=max_rf_features,
                                       threshold=-np.inf)
            selector.fit(X_corr, y)
            X_selected = selector.transform(X_corr)
            logger.info(f"After RF selection: {X_selected.shape}")
        
            return X_selected, y, scaler, selector

        X_selected, y_selected, scaler, selector = feature_selection(
            processed_df.drop(columns=['expt']).values,
            processed_df['expt'].values,
            corr_thresh=0.9,
            max_rf_features=mf,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=rs)
        
        def rf_cv(n_estimators, min_samples_split, min_samples_leaf, max_features_bo, max_depth):
            n_estimators = int(n_estimators)
            min_samples_split = int(min_samples_split)
            min_samples_leaf = int(min_samples_leaf)
            max_features_bo = round(max_features_bo, 2)
            max_depth = int(max_depth) if max_depth > 0 else None
            if max_features_bo <= 0:
                max_features_bo = 0.1
            elif max_features_bo > 1:
                max_features_bo = 1.0

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features_bo,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            return cv_scores.mean()

        pbounds = {
            'n_estimators': (10, 1000),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
            'max_features_bo': (0.1, 1.0),
            'max_depth': (1, 20)
        }

        optimizer = BayesianOptimization(
            f=rf_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=10, n_iter=100)

        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        best_params['max_features_bo'] = round(best_params['max_features_bo'], 2)
        best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] > 0 else None

        rf_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features_bo'],
            max_depth=best_params['max_depth'],
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # 存储结果
        all_results.append({
            'max_features': mf,
            'random_state': rs,
            'best_cv_r2': optimizer.max['target'],
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'best_params': best_params
        })
        
        # 存储预测数据
        for i in range(len(y_train)):
            all_prediction_data.append({
                'max_features': mf,
                'random_state': rs,
                'dataset': 'train',
                'index': i,
                'experimental': y_train[i],
                'predicted': y_train_pred[i]
            })
        for i in range(len(y_test)):
            all_prediction_data.append({
                'max_features': mf,
                'random_state': rs,
                'dataset': 'test',
                'index': i,
                'experimental': y_test[i],
                'predicted': y_test_pred[i]
            })
        
        # 记录每个max_features的最佳模型
        if mf not in best_model_per_mf or test_r2 > best_model_per_mf[mf]['test_r2']:
            best_model_per_mf[mf] = {
                'max_features': mf,
                'random_state': rs,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'best_params': best_params
            }

        with open(result_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([mf, rs, optimizer.max['target'], test_r2, test_mae, test_rmse])

        logger.info(f'Done. Test R²={test_r2:.4f}, Test MAE={test_mae:.4f}, Test RMSE={test_rmse:.4f}')

logger.info('All combinations finished.')

# ========== 1. 保存原始顺序的结果（按max_features从小到大，random_state从1-10） ==========
logger.info("\n【保存原始顺序结果】")
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(['max_features', 'random_state']).reset_index(drop=True)
results_df.to_csv('results_opt_morgan_1024m1.csv', index=False, float_format='%.6f')
logger.info(f"原始顺序结果已保存到: results_opt_morgan_1024m1.csv")

# ========== 2. 保存每个max_features的统计信息 ==========
logger.info("\n【保存每个max_features的统计信息】")
summary_data = []
for mf in max_features_list:
    mf_data = results_df[results_df['max_features'] == mf]
    best_idx = mf_data['test_r2'].idxmax()
    best_row = mf_data.loc[best_idx]
    summary_data.append({
        'max_features': mf,
        'mean_test_r2': mf_data['test_r2'].mean(),
        'std_test_r2': mf_data['test_r2'].std(),
        'mean_test_mae': mf_data['test_mae'].mean(),
        'mean_test_rmse': mf_data['test_rmse'].mean(),
        'best_test_r2': best_row['test_r2'],
        'best_random_state': best_row['random_state'],
        'best_test_mae': best_row['test_mae'],
        'best_test_rmse': best_row['test_rmse']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results_opt_morgan_1024m1_sorted.csv', index=False, float_format='%.6f')
logger.info(f"每个max_features的统计结果已保存到: results_opt_morgan_1024m1_sorted.csv")

# ========== 3. 保存最佳超参数组合 ==========
logger.info("\n【保存最佳超参数组合】")
best_params_list = []
for result in all_results:
    best_params_list.append({
        'max_features': result['max_features'],
        'random_state': result['random_state'],
        'n_estimators': result['best_params']['n_estimators'],
        'min_samples_split': result['best_params']['min_samples_split'],
        'min_samples_leaf': result['best_params']['min_samples_leaf'],
        'max_features_bo': result['best_params']['max_features_bo'],
        'max_depth': result['best_params']['max_depth'] if result['best_params']['max_depth'] is not None else 'None'
    })

best_params_df = pd.DataFrame(best_params_list)
best_params_df = best_params_df.sort_values(['max_features', 'random_state']).reset_index(drop=True)
best_params_df.to_csv('best_hyperparameters_morgan_1024m1.csv', index=False)
logger.info(f"最佳超参数组合已保存到: best_hyperparameters_morgan_1024m1.csv")

# ========== 4. 保存所有预测数据 ==========
logger.info("\n【保存所有预测数据】")
prediction_df = pd.DataFrame(all_prediction_data)
prediction_df = prediction_df.sort_values(['max_features', 'random_state', 'dataset', 'index']).reset_index(drop=True)
prediction_df.to_csv('prediction_data_morgan_1024m1.csv', index=False, float_format='%.6f')
logger.info(f"预测数据已保存到: prediction_data_morgan_1024m1.csv")
logger.info(f"总记录数: {len(prediction_df)}")

# ========== 5. 绘制每个max_features最佳模型的散点图 ==========
logger.info("\n【绘制每个max_features最佳模型的散点图】")

for mf in max_features_list:
    best = best_model_per_mf[mf]
    logger.info(f"绘制 max_features={mf} 的散点图 (best random_state={best['random_state']}, test_r2={best['test_r2']:.4f})")
    
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
    test_mae = best['test_mae']
    test_rmse = best['test_rmse']
    
    # 绘制完美预测线（黑色直线）
    all_y = np.concatenate([best['y_train'], best['y_test']])
    all_pred = np.concatenate([best['y_train_pred'], best['y_test_pred']])
    min_val = min(all_y.min(), all_pred.min())
    max_val = max(all_y.max(), all_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2, label='Perfect Prediction')
    
    # 设置坐标轴
    ax.set_xlabel('Experimental ΔEst (eV)', fontsize=14)
    ax.set_ylabel('Predicted ΔEst (eV)', fontsize=14)
    ax.set_title(f'Morgan Fingerprint (1024m1) - max_features={mf}, best random_state={best["random_state"]}\nRF Model Performance for ΔEst Prediction', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 创建图例（包含性能指标）
    train_legend = mlines.Line2D([], [], color='lightblue', marker='^', linestyle='None',
                                  markersize=10, label=f'Training: R²={train_r2:.3f}, MAE={train_mae:.3f}, RMSE={train_rmse:.3f}')
    test_legend = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                                 markersize=10, label=f'Test: R²={test_r2:.3f}, MAE={test_mae:.3f}, RMSE={test_rmse:.3f}')
    perfect_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=2, label='Perfect Prediction')
    
    # 添加图例
    ax.legend(handles=[train_legend, test_legend, perfect_line], loc='lower right', fontsize=10)
    
    # 设置等比例坐标轴
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val - 0.1, max_val + 0.1)
    ax.set_ylim(min_val - 0.1, max_val + 0.1)
    
    plt.tight_layout()
    plt.savefig(f'scatter_plot_morgan_1024m1_mf{mf}.png', dpi=300)
    plt.close()
    
    logger.info(f"  散点图已保存到: scatter_plot_morgan_1024m1_mf{mf}.png")

# ========== 6. 输出整体统计信息 ==========
logger.info("\n" + "="*60)
logger.info("整体统计信息 (Morgan指纹 1024m1)")
logger.info("="*60)
logger.info(f"总实验次数: {len(all_results)}")
logger.info(f"最佳测试集R²: {results_df['test_r2'].max():.4f}")
logger.info(f"平均测试集R²: {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}")
logger.info(f"平均测试集MAE: {results_df['test_mae'].mean():.4f}")
logger.info(f"平均测试集RMSE: {results_df['test_rmse'].mean():.4f}")

logger.info('\nAll tasks finished! (Morgan Fingerprint 1024m1)')