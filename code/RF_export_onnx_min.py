# -*- coding: utf-8 -*-
# 文件名建议：rf_export_onnx_min.py
# 功能：读取 Excel 的多个工作表，训练随机森林回归，并导出为 ONNX（用于网页端推理）
# 说明：这是“最小可用版”，已删除：参数搜索、SHAP、绘图、训练/测试指标输出、日志文件写入等内容。

import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ================== 你只需要改这 4 项 ==================
EXCEL_PATH = r"D:\Users\HUAWEI\Desktop\Age_Estimation_Project\female.xlsx"    # 你的 Excel
OUT_DIR    = r"D:\Users\HUAWEI\Desktop\Age_Estimation_Project\models\RFfemale" # 输出目录（存 .onnx）

AGE_COL_IDX = 2     # 年龄列索引：如果你的年龄在 C 列 -> 2（A=0,B=1,C=2）
X_START     = 11    # 特征起始列索引（左闭）
X_END       = 21    # 特征结束列索引（右开）=> X_START: X_END 应该刚好 10 列
# ======================================================

# 固定 10 个特征名（用于保存给网页端做顺序对齐）
FEATURE_NAMES = [chr(ord('M') + i) for i in range(10)]  # ['M'...'V']

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
                # 允许负号和一个小数点
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

        X.columns = FEATURE_NAMES

        # 训练（为导出模型：这里直接用全量数据训练，不再分训练/测试集）
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # 导出 ONNX
        safe = safe_filename(sheet)
        onnx_path = os.path.join(OUT_DIR, f"rf_{safe}.onnx")
        initial_type = [("input", FloatTensorType([None, X.shape[1]]))]  # [None,10]
        onx = convert_sklearn(rf, initial_types=initial_type, target_opset=17)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        # 保存特征顺序（网页端构造输入向量必须用同顺序）
        feat_path = os.path.join(OUT_DIR, f"rf_{safe}.features.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)

        print(f"导出完成：{onnx_path}")
        print(f"特征顺序：{feat_path}")

    print("\n全部工作表处理完毕。")

if __name__ == "__main__":
    main()
