# -*- coding: utf-8 -*-
"""
Morgan指纹描述符超参数优化 - XGBoost模型（带早停）
使用特征重要性排序后的数据
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
import csv
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xgboost as xgb

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
file0 = Path('Est_1024m1_cleaned_sorted.pkl')
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
    X_full = processed_df.drop(columns=['expt']).values
    y = processed_df['expt'].values
    feature_names = processed_df.columns[:-1].tolist()

logger.info(f"数据形状: X={X_full.shape}, y={y.shape}")
logger.info(f"特征数量: {len(feature_names)}")

# ========== 数据清洗 ==========
float32_max = np.finfo(np.float32).max
print("\n数据清洗...")
print(f"原始 X_full 中无穷大数量: {np.isinf(X_full).sum()}")
print(f"原始 X_full 中NaN数量: {np.isnan(X_full).sum()}")

X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
X_full = np.clip(X_full, -float32_max, float32_max)
X_full = X_full.astype(np.float64)
y = y.astype(np.float64)

print(f"清洗后 X_full 中无穷大数量: {np.isinf(X_full).sum()}")
print(f"清洗后 X_full 中NaN数量: {np.isnan(X_full).sum()}")
# =============================

# 2. 准备结果文件
result_csv = 'results_opt_morgan_1024m1_xgb_earlystop.csv'
with open(result_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['max_features', 'random_state', 'best_cv_r2', 'test_r2', 'test_mae', 'test_rmse', 'actual_n_estimators'])

# 3. 网格搜索参数
max_features_list = [20, 50, 60, 100, 1026]
random_state_list = list(range(1, 11))

# 存储所有结果
all_results = []
all_prediction_data = []
best_model_per_mf = {}

for mf in max_features_list:
    for rs in random_state_list:
        logger.info(f'=== Running max_features={mf}, random_state={rs} ===')

        # 选择前mf个最重要的特征
        selected_idx = list(range(min(mf, len(feature_names))))
        X_selected = X_full[:, selected_idx]
        y_selected = y

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_selected, test_size=0.2, random_state=rs
        )
        
        # 从训练集中再划分验证集（用于早停）
        X_train_cv, X_val, y_train_cv, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # XGBoost交叉验证函数
        def xgb_cv(n_estimators, max_depth, learning_rate, subsample, 
                   colsample_bytree, min_child_weight, reg_alpha, reg_lambda):
            n_estimators = int(n_estimators)
            max_depth = int(max_depth)
            learning_rate = round(learning_rate, 3)
            subsample = round(subsample, 2)
            colsample_bytree = round(colsample_bytree, 2)
            min_child_weight = round(min_child_weight, 1)
            reg_alpha = round(reg_alpha, 3)
            reg_lambda = round(reg_lambda, 3)
            
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=-1
            )
            cv_scores = cross_val_score(model, X_train_cv, y_train_cv, cv=5, scoring='r2')
            return cv_scores.mean()

        # XGBoost超参数空间
        pbounds = {
            'n_estimators': (10, 1000),
            'max_depth': (1, 20),
            'learning_rate': (0.01, 0.5),
            'subsample': (0.2, 1.0),
            'colsample_bytree': (0.2, 1.0),
            'min_child_weight': (1, 20),
            'reg_alpha': (0, 5),
            'reg_lambda': (0.1, 10)
        }

        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=10, n_iter=100)

        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['learning_rate'] = round(best_params['learning_rate'], 3)
        best_params['subsample'] = round(best_params['subsample'], 2)
        best_params['colsample_bytree'] = round(best_params['colsample_bytree'], 2)
        best_params['min_child_weight'] = round(best_params['min_child_weight'], 1)
        best_params['reg_alpha'] = round(best_params['reg_alpha'], 3)
        best_params['reg_lambda'] = round(best_params['reg_lambda'], 3)

        # 训练早停模型确定最佳树数量
        xgb_model = xgb.XGBRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            min_child_weight=best_params['min_child_weight'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            early_stopping_rounds=30,
            eval_metric='rmse',
            random_state=42,
            n_jobs=-1
        )
        
        eval_set = [(X_val, y_val)]
        xgb_model.fit(X_train_cv, y_train_cv, eval_set=eval_set, verbose=False)
        
        # 获取早停后实际使用的树数量
        actual_n_estimators = xgb_model.best_iteration
        if actual_n_estimators is None:
            actual_n_estimators = best_params['n_estimators']
        
        # 用实际树数量在完整训练集上重新训练
        xgb_model_final = xgb.XGBRegressor(
            n_estimators=actual_n_estimators,
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            min_child_weight=best_params['min_child_weight'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            random_state=42,
            n_jobs=-1
        )
        xgb_model_final.fit(X_train, y_train)

        y_train_pred = xgb_model_final.predict(X_train)
        y_test_pred = xgb_model_final.predict(X_test)
        
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
            'best_params': best_params,
            'actual_n_estimators': actual_n_estimators
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
                'best_params': best_params,
                'actual_n_estimators': actual_n_estimators
            }

        with open(result_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([mf, rs, optimizer.max['target'], test_r2, test_mae, test_rmse, actual_n_estimators])

        logger.info(f'Done. Test R²={test_r2:.4f}, Test MAE={test_mae:.6f} eV, '
                    f'Test RMSE={test_rmse:.6f} eV, Actual trees={actual_n_estimators}')

logger.info('All combinations finished.')

# ========== 后续保存结果、绘制散点图等 ==========
logger.info("\n【保存原始顺序结果】")
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(['max_features', 'random_state']).reset_index(drop=True)
results_df.to_csv('results_opt_morgan_1024m1_xgb_earlystop.csv', index=False, float_format='%.6f')
logger.info(f"原始顺序结果已保存到: results_opt_morgan_1024m1_xgb_earlystop.csv")

logger.info("\n【保存每个max_features的统计信息】")
summary_data = []
for mf in max_features_list:
    mf_data = results_df[results_df['max_features'] == mf]
    if len(mf_data) > 0:
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
            'best_test_rmse': best_row['test_rmse'],
            'best_actual_n_estimators': best_row['actual_n_estimators']
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results_opt_morgan_1024m1_xgb_earlystop_sorted.csv', index=False, float_format='%.6f')
logger.info(f"每个max_features的统计结果已保存到: results_opt_morgan_1024m1_xgb_earlystop_sorted.csv")

logger.info("\n【保存最佳超参数组合】")
best_params_list = []
for result in all_results:
    best_params_list.append({
        'max_features': result['max_features'],
        'random_state': result['random_state'],
        'n_estimators': result['best_params']['n_estimators'],
        'actual_n_estimators': result['actual_n_estimators'],
        'max_depth': result['best_params']['max_depth'],
        'learning_rate': result['best_params']['learning_rate'],
        'subsample': result['best_params']['subsample'],
        'colsample_bytree': result['best_params']['colsample_bytree'],
        'min_child_weight': result['best_params']['min_child_weight'],
        'reg_alpha': result['best_params']['reg_alpha'],
        'reg_lambda': result['best_params']['reg_lambda']
    })

best_params_df = pd.DataFrame(best_params_list)
best_params_df = best_params_df.sort_values(['max_features', 'random_state']).reset_index(drop=True)
best_params_df.to_csv('best_hyperparameters_morgan_1024m1_xgb_earlystop.csv', index=False)
logger.info(f"最佳超参数组合已保存到: best_hyperparameters_morgan_1024m1_xgb_earlystop.csv")

logger.info("\n【保存所有预测数据】")
prediction_df = pd.DataFrame(all_prediction_data)
prediction_df = prediction_df.sort_values(['max_features', 'random_state', 'dataset', 'index']).reset_index(drop=True)
prediction_df.to_csv('prediction_data_morgan_1024m1_xgb_earlystop.csv', index=False, float_format='%.6f')
logger.info(f"预测数据已保存到: prediction_data_morgan_1024m1_xgb_earlystop.csv")
logger.info(f"总记录数: {len(prediction_df)}")

# ========== 绘制散点图 ==========
logger.info("\n【绘制每个max_features最佳模型的散点图】")

for mf in max_features_list:
    if mf not in best_model_per_mf:
        continue
    best = best_model_per_mf[mf]
    logger.info(f"绘制 max_features={mf} 的散点图 (best random_state={best['random_state']}, test_r2={best['test_r2']:.4f})")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(best['y_train'], best['y_train_pred'], alpha=0.6, s=50, 
               c='lightblue', marker='^', edgecolors='steelblue', linewidth=1, label='Training Set')
    
    ax.scatter(best['y_test'], best['y_test_pred'], alpha=0.6, s=50, 
               c='orange', marker='o', edgecolors='darkorange', linewidth=1, label='Test Set')
    
    train_r2 = r2_score(best['y_train'], best['y_train_pred'])
    train_mae = mean_absolute_error(best['y_train'], best['y_train_pred'])
    train_rmse = np.sqrt(mean_squared_error(best['y_train'], best['y_train_pred']))
    
    test_r2 = best['test_r2']
    test_mae = best['test_mae']
    test_rmse = best['test_rmse']
    
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
    
    ax.set_xlabel('Experimental ΔEst (eV)', fontsize=14)
    ax.set_ylabel('Predicted ΔEst (eV)', fontsize=14)
    ax.set_title(f'Morgan Fingerprint (1024m1 - XGBoost + EarlyStop) - max_features={mf}, best random_state={best["random_state"]}\nModel Performance for ΔEst Prediction', fontsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    train_legend = mlines.Line2D([], [], color='lightblue', marker='^', linestyle='None',
                                  markersize=10, label=f'Training: R²={train_r2:.3f}, MAE={train_mae:.3f}, RMSE={train_rmse:.3f}')
    test_legend = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                                 markersize=10, label=f'Test: R²={test_r2:.3f}, MAE={test_mae:.3f}, RMSE={test_rmse:.3f}')
    perfect_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=2, label='Perfect Prediction')
    
    ax.legend(handles=[train_legend, test_legend, perfect_line], loc='lower right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f'scatter_plot_morgan_1024m1_xgb_earlystop_mf{mf}.png', dpi=300)
    plt.close()
    
    logger.info(f"  散点图已保存到: scatter_plot_morgan_1024m1_xgb_earlystop_mf{mf}.png")

# ========== 输出整体统计信息 ==========
logger.info("\n" + "="*60)
logger.info("整体统计信息 (Morgan指纹 1024m1 - XGBoost + EarlyStop)")
logger.info("="*60)
logger.info(f"总实验次数: {len(all_results)}")
logger.info(f"最佳测试集R²: {results_df['test_r2'].max():.4f}")
logger.info(f"平均测试集R²: {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}")
logger.info(f"平均测试集MAE: {results_df['test_mae'].mean():.6f} eV")
logger.info(f"平均测试集RMSE: {results_df['test_rmse'].mean():.6f} eV")

logger.info('\nAll tasks finished! (Morgan Fingerprint 1024m1 - XGBoost + EarlyStop)')