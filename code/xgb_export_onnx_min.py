# -*- coding: utf-8 -*-
# 文件名建议：xgb_export_onnx_min.py
# 功能：读取 Excel 的多个工作表，训练 XGBoost 回归，并导出为 ONNX（用于网页端推理）

import os
import json
import pandas as pd
import xgboost as xgb
import onnxmltools
# 使用 onnxconverter-common（已安装）
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

import onnx

# ================== 你只需要改这 4 项 ==================
EXCEL_PATH = r"D:\Users\HUAWEI\Desktop\Age_Estimation_Project\female.xlsx"
OUT_DIR    = r"D:\Users\HUAWEI\Desktop\Age_Estimation_Project\models\XGBoostfemale"

AGE_COL_IDX = 2     # 年龄列索引
X_START     = 11    # 特征起始列索引
X_END       = 21    # 特征结束列索引
# ======================================================

# 原始特征名（字母格式）
ORIGINAL_FEATURE_NAMES = [chr(ord('M') + i) for i in range(10)]  # ['M'...'V']
# ONNX 兼容的特征名（数字格式）
ONNX_FEATURE_NAMES = [f'f{i}' for i in range(10)]  # ['f0'...'f9']

def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(name)).strip().replace(" ", "_")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xls = pd.ExcelFile(EXCEL_PATH)
    sheets = xls.sheet_names
    print(f"检测到 {len(sheets)} 个工作表：{sheets}")

    for sheet in sheets:
        print("\n" + "="*60)
        print(f"处理工作表：{sheet}")

        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=None)

        # 若第一行是表头（包含非数字），就去掉第一行
        if df.shape[0] > 0:
            first_row = df.iloc[0].astype(str).tolist()
            def _is_numberish(s: str) -> bool:
                s = s.strip()
                if s == "":
                    return False
                if s.startswith("-"):
                    s = s[1:]
                parts = s.split(".")
                return len(parts) <= 2 and all(p.isdigit() for p in parts if p != "")
            if any(not _is_numberish(s) for s in first_row):
                df = df.iloc[1:].reset_index(drop=True)

        # 转数值、去缺失
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty or len(df) < 10:
            print("数据不足/无效，跳过。")
            continue

        # 取 y（年龄）
        if df.shape[1] <= AGE_COL_IDX:
            raise ValueError(f"年龄列索引 AGE_COL_IDX={AGE_COL_IDX} 超出范围：你的表只有 {df.shape[1]} 列")
        y = df.iloc[:, AGE_COL_IDX].astype(float)

        # 取 X（10 个特征）
        if df.shape[1] < X_END:
            raise ValueError(f"特征列范围 [{X_START}:{X_END}] 超出范围：你的表只有 {df.shape[1]} 列")
        X = df.iloc[:, X_START:X_END].astype(float)

        if X.shape[1] != 10:
            raise ValueError(
                f"特征列数量不等于 10：当前 X.shape={X.shape}。"
                f"请检查 X_START={X_START}, X_END={X_END} 是否正确。"
            )

        # 关键修改：使用数字格式的特征名
        X.columns = ONNX_FEATURE_NAMES

        # 训练（为导出模型：这里直接用全量数据训练）
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        # 导出 ONNX - 使用 onnxmltools
        # booster = model.get_booster()

        safe = safe_filename(sheet)
        onnx_path = os.path.join(OUT_DIR, f"xgb_{safe}.onnx")

        try:
            # 尝试使用 opset 15
            onnx_model = convert_xgboost(
                model, 
                initial_types=[('input', FloatTensorType([None, X.shape[1]]))], 
                target_opset=15
            )
        except Exception as e:
            print(f"opset 15 失败: {e}")
            try:
                # 尝试不使用 target_opset 参数
                onnx_model = convert_xgboost(
                    model, 
                    initial_types=[('input', FloatTensorType([None, X.shape[1]]))]
                )
            except Exception as e2:
                print(f"转换失败: {e2}")
                continue
        
        # 保存 ONNX 模型
        onnxmltools.utils.save_model(onnx_model, onnx_path)

        # 保存特征顺序映射（原始特征名 -> ONNX特征名）
        # 这样网页端就知道如何映射输入
        feature_mapping = {
            "original_features": ORIGINAL_FEATURE_NAMES,
            "onnx_features": ONNX_FEATURE_NAMES,
            "mapping": dict(zip(ORIGINAL_FEATURE_NAMES, ONNX_FEATURE_NAMES))
        }
        
        feat_path = os.path.join(OUT_DIR, f"xgb_{safe}.features.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feature_mapping, f, ensure_ascii=False, indent=2)

        print(f"导出完成：{onnx_path}")
        print(f"特征映射：{feat_path}")
        print(f"原始特征名: {ORIGINAL_FEATURE_NAMES}")
        print(f"ONNX特征名: {ONNX_FEATURE_NAMES}")

    print("\n全部工作表处理完毕。")

if __name__ == "__main__":
    main()