# 文件名建议保存为：age_prediction.py
# 功能：读取 Excel 多个工作表，用 M-V 列预测 C 列（年龄），随机森林回归 + 新增可视化

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告

# === 配置区：请确保路径正确 ==
EXCEL_PATH = r"D:\Users\HUAWEI\Desktop\FS\female.xlsx"
OUTPUT_DIR = r"D:\Users\HUAWEI\Desktop\FS"  # 新增图表输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

# ==============================
# 配置日志
log_file = r"D:\Users\HUAWEI\Desktop\FS\femaletest.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

def log_and_print(message):
    """同时打印到控制台和记录到日志"""
    print(message)
    logging.info(message)

def safe_filename(name):
    """移除文件名中的非法字符"""
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in str(name))

def plot_feature_importance(model, feature_names, sheet_name, output_dir):
    """绘制特征重要性条形图"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"特征重要性 - {sheet_name}", fontsize=14)
    plt.bar(range(len(importances)), importances[indices], color="b", align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    safe_name = safe_filename(sheet_name)
    plt.savefig(os.path.join(output_dir, f"特征重要性_{safe_name}.png"), dpi=300)
    plt.close()

def plot_error_distribution(y_true, y_pred, sheet_name, output_dir):
    """绘制预测误差分布图"""
    errors = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'平均误差: {np.mean(errors):.2f}')
    plt.xlabel("预测误差 (实际年龄 - 预测年龄)", fontsize=12)
    plt.ylabel("样本数量", fontsize=12)
    plt.title(f"预测误差分布 - {sheet_name}", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    safe_name = safe_filename(sheet_name)
    plt.savefig(os.path.join(output_dir, f"误差分布_{safe_name}.png"), dpi=300)
    plt.close()

def plot_shap_summary(model, X_sample, feature_names, sheet_name, output_dir):
    """绘制SHAP摘要图和蜂群图"""
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP 特征重要性摘要 - {sheet_name}", fontsize=14)
    safe_name = safe_filename(sheet_name)
    plt.savefig(os.path.join(output_dir, f"SHAP摘要_{safe_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP蜂群图 (Violin Plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP 蜂群图 - {sheet_name}", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"SHAP蜂群图_{safe_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # 读取所有工作表
        xls = pd.ExcelFile(EXCEL_PATH)
        sheet_names = xls.sheet_names
        log_and_print(f"检测到 {len(sheet_names)} 个工作表: {sheet_names}\n")

        # 定义特征名称 (M到V列)
        feature_names = [chr(ord('M') + i) for i in range(10)]  # 生成 ['M', 'N', 'O', ..., 'V']
        
        for sheet in sheet_names:
            log_and_print(f"{'='*50}")
            log_and_print(f"处理工作表: {sheet}")
            log_and_print(f"{'='*50}")
            
            # 读取数据
            df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=None)
            
            # 自动跳过表头行（如果第一行包含 'number' 或 'age'）
            if df.shape[0] > 0 and ('number' in str(df.iloc[0, 0]) or 'age' in str(df.iloc[0, 2])):
                df = df.iloc[1:].reset_index(drop=True)
            
            # 转为数值并清理
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            if df.empty or len(df) < 5:
                log_and_print("  ⚠️ 数据不足或无效，跳过。\n")
                continue

            # 提取年龄（第3列，C列，索引2）
            y = df.iloc[:, 1]  # 注意：这里修正为索引2（C列）
            # 提取特征（M到V列：第13~22列，索引12~21）
            X = df.iloc[:, 11:21]  # 修正为索引12-21（共10列）
            X.columns = feature_names  # 设置特征名称

            if X.shape[1] != 10:
                log_and_print(f"  ⚠️ 特征列数量错误（应为10列，实际{X.shape[1]}），跳过。\n")
                continue

            # 替换原有的数据集划分代码
            # 划分数据集为训练集、验证集和测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of original data
            )

            log_and_print(f"  📊 数据集划分: 训练集({len(X_train)}), 验证集({len(X_val)}), 测试集({len(X_test)})")
            
            # 替换原有的参数测试部分为以下代码
            def cross_validate_with_regularization(X, y):
                """使用正则化参数进行交叉验证"""
                
                # 参数网格（更注重防止过拟合）
                param_grid = {
                    'n_estimators': [50, 100, 150],           # 增加一个中间值
                    'max_features': ['sqrt', 'log2', 2, 3, 4], # 添加log2选项
                    'max_depth': [3, 5, 8, None],             # 添加不限制深度的选项
                    'min_samples_split': [10, 15, 20, 25],    # 增加一个更大值
                    'min_samples_leaf': [5, 8, 10, 15],       # 增加一个更大值
                   'bootstrap': [True, False],               # 添加是否放回抽样选项
                    'max_samples': [0.6, 0.8, 1.0]           # 添加采样比例（仅当bootstrap=True时有效）
                 }
                
                best_mae = np.inf
                best_params = {}
                best_r2 = -np.inf
                
                log_and_print(f"  📋 测试参数组合...")
                
                # 简化版网格搜索
                for n_est in param_grid['n_estimators'][:2]:  # 只取前两个值
                    for max_feat in param_grid['max_features']:
                        for max_dep in param_grid['max_depth'][:2]:  # 只取前两个值
                            for min_ss in param_grid['min_samples_split']:
                                for min_sl in param_grid['min_samples_leaf']:
                                    
                                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                                    fold_mae_scores = []
                                    fold_r2_scores = []
                                    
                                    # 进行5折交叉验证
                                    for train_index, val_index in kfold.split(X):
                                        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                                        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
                                        
                                        # 创建模型
                                        model = RandomForestRegressor(
                                            n_estimators=n_est,
                                            max_features=max_feat,
                                            max_depth=max_dep,
                                            min_samples_split=min_ss,
                                            min_samples_leaf=min_sl,
                                            random_state=42,
                                            n_jobs=1
                                        )
                                        
                                        model.fit(X_train_fold, y_train_fold)
                                        
                                        # 预测与评估
                                        y_pred = model.predict(X_val_fold)
                                        fold_mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
                                        fold_r2_scores.append(r2_score(y_val_fold, y_pred))
                                    
                                    # 计算平均分数
                                    avg_mae = np.mean(fold_mae_scores)
                                    avg_r2 = np.mean(fold_r2_scores)
                                    
                                    # 更新最佳参数（以MAE为准）
                                    if avg_mae < best_mae:
                                        best_mae = avg_mae
                                        best_r2 = avg_r2
                                        best_params = {
                                            'n_estimators': n_est,
                                            'max_features': max_feat,
                                            'max_depth': max_dep,
                                            'min_samples_split': min_ss,
                                            'min_samples_leaf': min_sl
                                        }
                
                return best_params, best_r2, best_mae

            # 在主循环中使用这个函数
            log_and_print("  🔍 开始参数搜索（防止过拟合优化）...")
            best_params, best_r2, best_mae = cross_validate_with_regularization(X, y)

            log_and_print(f"\n  🏆 最佳参数组合:")
            for key, value in best_params.items():
                log_and_print(f"    {key}: {value}")
            log_and_print(f"    最佳 R² Score: {best_r2:.4f}")
            log_and_print(f"    对应 MAE: {best_mae:.2f} 岁")

            # 使用最佳参数重新训练模型（修改这部分）
            best_model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_features=best_params['max_features'],
                max_depth=best_params.get('max_depth', 10),  # 限制树的深度
                min_samples_split=best_params.get('min_samples_split', 10),  # 增加分裂所需最小样本数
                min_samples_leaf=best_params.get('min_samples_leaf', 4),  # 增加叶节点最小样本数
                random_state=42,
                n_jobs=-1
            )
            best_model.fit(X_train, y_train)
            
            # ======== 新增：计算训练集和测试集R² ========
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            log_and_print(f"  ✅ 样本总数: {len(df)}")
            log_and_print(f"  📊 训练集 R² Score: {train_r2:.4f}")
            log_and_print(f"  📊 测试集 R² Score: {test_r2:.4f}")
            log_and_print(f"  📏 测试集 MAE (平均绝对误差): {test_mae:.2f} 岁\n")
            
            # ======== 新增：生成可视化图表 ========
            safe_sheet_name = safe_filename(sheet)
            sheet_output_dir = os.path.join(OUTPUT_DIR, safe_sheet_name)
            os.makedirs(sheet_output_dir, exist_ok=True)
            
            log_and_print(f"  📈 正在生成可视化图表到: {sheet_output_dir}")
            
            # 1. 特征重要性图
            plot_feature_importance(best_model, feature_names, sheet, sheet_output_dir)
            
            # 2. 误差分布图 (使用测试集)
            plot_error_distribution(y_test, y_test_pred, sheet, sheet_output_dir)
            
            # 3. SHAP分析 (使用训练集的子集加速计算)
            sample_size = min(1000, len(X_train))  # 限制SHAP计算的样本量
            X_sample = X_train.sample(n=sample_size, random_state=42)
            plot_shap_summary(best_model, X_sample, feature_names, sheet, sheet_output_dir)
            
            log_and_print(f"  ✅ 可视化图表生成完成")
            log_and_print(f"    - 特征重要性图: 特征重要性_{safe_sheet_name}.png")
            log_and_print(f"    - 误差分布图: 误差分布_{safe_sheet_name}.png")
            log_and_print(f"    - SHAP摘要图: SHAP摘要_{safe_sheet_name}.png")
            log_and_print(f"    - SHAP蜂群图: SHAP蜂群图_{safe_sheet_name}.png\n")

    except FileNotFoundError:
        log_and_print(f"❌ 错误：找不到文件！请检查路径是否正确：\n{EXCEL_PATH}")
    except Exception as e:
        log_and_print(f"💥 程序出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    log_and_print("✅ 所有工作表处理完毕！")