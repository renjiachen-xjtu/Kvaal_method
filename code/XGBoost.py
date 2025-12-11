# 1. 安装必要库（如果还没安装）
# 在终端或命令行运行（只需一次）：
# pip install pandas scikit-learn xgboost matplotlib seaborn shap

# 2. 导入所需库
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')  # 忽略所有警告

# 3. 读取 Excel 文件（所有工作表）
excel_file = pd.read_excel(r"D:\Users\HUAWEI\Desktop\1\female.xlsx", sheet_name=None)

# === 关键修改: 设置统一的输出目录结构 ===
MAIN_OUTPUT_DIR = r"D:\Users\HUAWEI\Desktop\1"  # 主输出目录
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)  # 确保主目录存在

# 日志文件路径
output_file_path = os.path.join(MAIN_OUTPUT_DIR, "testmale.txt")

# 初始化日志（带时间戳）
with open(output_file_path, 'a', encoding='utf-8') as f:
    f.write("\n" + "="*60 + "\n")
    f.write(f"【新运行开始】时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"共检测到 {len(excel_file)} 个工作表: {list(excel_file.keys())}\n")
    f.write("="*60 + "\n")

# 4. 指定特征列
feature_columns = [
    "L=(P+R)/2",
    "M=(P+R+A+B+C)/5",
    "W=(B+C/2)",
    "W-L",
    "P=p/r",
    "R=p/t",
    "T=t/r",
    "A=a`/a",
    "B=b`/b",
    "C=c`/c"
]

# 安全文件名函数（与随机森林代码一致）
def safe_filename(name):
    """移除文件名中的非法字符"""
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in str(name))

# === 新增：绘制特征重要性条形图 ===
def plot_feature_importance(model, feature_names, sheet_name, output_dir):
    """绘制特征重要性条形图"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 从大到小排序的索引
    
    plt.figure(figsize=(10, 6))
    plt.title(f"特征重要性 - {sheet_name}", fontsize=14)
    plt.bar(range(len(importances)), importances[indices], color="b", align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    safe_name = safe_filename(sheet_name)
    plt.savefig(os.path.join(output_dir, f"特征重要性_{safe_name}.png"), dpi=300)
    plt.close()

# === 新增：绘制误差分布图 ===
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

# === 新增：绘制SHAP摘要图和蜂群图 ===
def plot_shap_summary(model, X_sample, feature_names, sheet_name, output_dir):
    """绘制SHAP摘要图和蜂群图"""
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP摘要图 (条形图)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP 特征重要性摘要 - {sheet_name}", fontsize=14)
    safe_name = safe_filename(sheet_name)
    plt.savefig(os.path.join(output_dir, f"SHAP摘要_{safe_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP蜂群图 (Violin Plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="violin", show=False)
    plt.title(f"SHAP 蜂群图 - {sheet_name}", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"SHAP蜂群图_{safe_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 5. 对每个工作表独立建模分析
for sheet_name, df in excel_file.items():
    print(f"正在处理工作表: {sheet_name}")
    
    # === 为当前工作表创建专属目录 ===
    safe_sheet_name = safe_filename(sheet_name)
    sheet_output_dir = os.path.join(MAIN_OUTPUT_DIR, safe_sheet_name)
    os.makedirs(sheet_output_dir, exist_ok=True)
    print(f"  📁 创建/使用目录: {sheet_output_dir}")
    
    # 检查必要列是否存在
    missing_cols = [col for col in feature_columns + ["age"] if col not in df.columns]
    if missing_cols:
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n⚠️ 工作表 '{sheet_name}' 缺少列: {missing_cols}，跳过分析。\n")
        continue

    # 提取特征和目标变量
    X = df[feature_columns]
    y = df["age"]

    # 检查空值
    if X.isnull().any().any() or y.isnull().any():
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n⚠️ 工作表 '{sheet_name}' 包含缺失值，跳过分析。\n")
        continue

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 优化参数网格减少过拟合 ===
    param_grid = {
        'n_estimators': [100, 200, 300],  # 限制最大树数量
        'max_depth': [2, 3, 4],           # 严格限制树深度（小样本防过拟合）
        'learning_rate': [0.01, 0.05, 0.1]  # 降低学习率，增强泛化
    }

    # === 增强正则化参数 ===
    xgb_model = XGBRegressor(
        random_state=42,
        subsample=0.7,              # 减少样本采样比例
        colsample_bytree=0.7,       # 减少特征采样比例
        reg_alpha=0.3,              # 增加L1正则化
        reg_lambda=1.5,             # 增加L2正则化
        gamma=0.2,                  # 增加分裂所需最小损失减少
        n_jobs=1,                   # 避免多进程问题
        tree_method='hist'          # 使用直方图算法加速
    )

    # 网格搜索
    model = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=1,  # 单进程更稳定
        verbose=0
    )

    # 训练模型
    model.fit(X_train, y_train)
    best_mae = -model.best_score_
    best_params = model.best_params_

    # 预测
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)  # 训练集预测

    # === 计算四个核心指标 ===
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    # === 生成可视化图表 ===
    print(f"  📈 正在生成可视化图表到: {sheet_output_dir}")
    
    # 1. 特征重要性条形图
    plot_feature_importance(model.best_estimator_, feature_columns, sheet_name, sheet_output_dir)
    
    # 2. 误差分布图 (使用测试集)
    plot_error_distribution(y_test, y_pred, sheet_name, sheet_output_dir)
    
    # 3. SHAP分析 (使用训练集的子集加速计算)
    sample_size = min(1000, len(X_train))  # 限制SHAP计算的样本量
    X_sample = X_train.sample(n=sample_size, random_state=42)
    plot_shap_summary(model.best_estimator_, X_sample, feature_columns, sheet_name, sheet_output_dir)
    
    print(f"  ✅ 可视化图表生成完成")
    print(f"    - 特征重要性图: 特征重要性_{safe_sheet_name}.png")
    print(f"    - 误差分布图: 误差分布_{safe_sheet_name}.png")
    print(f"    - SHAP摘要图: SHAP摘要_{safe_sheet_name}.png")
    print(f"    - SHAP蜂群图: SHAP蜂群图_{safe_sheet_name}.png")

    # === 保存本工作表的完整结果到日志文件 ===
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"工作表 '{sheet_name}' 的完整分析结果\n")
        f.write(f"{'='*50}\n")
        
        f.write(f"图表保存目录: {sheet_output_dir}\n")
        f.write("-" * 30 + "\n")

        f.write("模型配置信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"最佳参数: {best_params}\n")
        f.write(f"参数网格: {param_grid}\n")
        f.write("增强正则化配置:\n")
        f.write("  subsample=0.7, colsample_bytree=0.7\n")
        f.write("  reg_alpha=0.3, reg_lambda=1.5, gamma=0.2\n")
        f.write(f"交叉验证折数: 5\n")

        f.write("\n模型性能指标:\n")
        f.write("-" * 30 + "\n")
        # === 添加四个核心指标 ===
        f.write(f"训练集 R²: {train_r2:.4f}\n")
        f.write(f"测试集 R²: {test_r2:.4f}\n")
        f.write(f"训练集 MAE: {train_mae:.2f} 岁\n")
        f.write(f"测试集 MAE: {test_mae:.2f} 岁\n")
        f.write(f"最佳交叉验证 MAE: {best_mae:.2f}\n")

        f.write("\n特征重要性排序:\n")
        f.write("-" * 30 + "\n")
        importances = model.best_estimator_.feature_importances_
        feature_importance = list(zip(feature_columns, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for name, importance in feature_importance:
            f.write(f"{name}: {importance:.4f}\n")

        f.write("\n数据集信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总样本数: {len(X)} (小样本注意过拟合风险)\n")
        f.write(f"训练集样本数: {len(X_train)}\n")
        f.write(f"测试集样本数: {len(X_test)}\n")
        f.write(f"特征数量: {len(feature_columns)}\n")

        f.write("\n预测结果对比 (前10个测试样本):\n")
        f.write("-" * 30 + "\n")
        f.write("真实年龄\t预测年龄\t误差\n")
        for i in range(min(10, len(y_test))):
            true_age = y_test.iloc[i]
            pred_age = y_pred[i]
            error = abs(true_age - pred_age)
            f.write(f"{true_age:.1f}\t\t{pred_age:.1f}\t\t{error:.1f}\n")

        f.write("\n使用的特征列:\n")
        f.write("-" * 30 + "\n")
        for i, col in enumerate(feature_columns, 1):
            f.write(f"{i}. {col}\n")

        f.write(f"\n生成图表 (保存在 {safe_sheet_name} 目录):\n")
        f.write(f"- 特征重要性图: 特征重要性_{safe_sheet_name}.png\n")
        f.write(f"- 误差分布图: 误差分布_{safe_sheet_name}.png\n")
        f.write(f"- SHAP摘要图: SHAP摘要_{safe_sheet_name}.png\n")
        f.write(f"- SHAP蜂群图: SHAP蜂群图_{safe_sheet_name}.png\n")
        f.write(f"{'='*50}\n")

    print(f"✅ 工作表 '{sheet_name}' 分析完成，结果已追加到日志文件。")

print(f"\n所有工作表分析完毕！详细结果已追加到: {output_file_path}")