import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 选择标签列和特征列
def select_columns(df):
    n = 0
    col_list = []
    for col in df.columns:
        print(f"{n}, {col}")
        col_list.append(col)
        n += 1
    while True:
        try:
            i = int(input("请选择标签列，及y："))
            if i >= n:
                print(f"请输入正确的数字(0-{n})!")
                continue
        except ValueError:
            print("请输入数字！")
            continue
        break
    y_series = df[col_list[i]]
    label_col = col_list[i]
    feature_cols = input("请输入特征列(空格分隔)：").split()
    select_names = [col_list[int(i)] for i in feature_cols]
    x_df = df[select_names]
    print(f"标签：{col_list[i]}， 特征：{select_names}共{len(select_names)}列")


    return x_df, y_series, feature_cols, label_col

# 线性缩放
def minmax_scale(series):
    x_stand = (series - series.min()) / (series.max() - series.min())
    return x_stand

# Z得分
def zscore_scale(series):
    x_stand = (series - series.mean()) / series.std()
    return x_stand

# 对数缩放
def log_scale(series):
    x_stand = np.log(series)
    return x_stand

# 调用函数
def apply_normalization(x_df):
    original = x_df.copy()
    # 找出所有数值列
    num_list = []
    for i in x_df.columns:
        if pd.api.types.is_numeric_dtype(x_df[i]):
            num_list.append(i)

    if not num_list:
        print("没有数值列，跳过归一化")
        return x_df
    
    # 选择数值列
    print("数值列：")
    for idx, col in enumerate(num_list):
        print(f"{idx}, {col}")
    selected = input("选择要归一化的列(空格分隔，直接回车=全部)：").split()

    if selected:
        target_cols = [num_list[int(i)] for i in selected]
    else:
        target_cols = num_list

    # 选择归一化方法
    while True:
        print("请选择归一化方法：\n[1] 线性缩放   [2] Z-score   [3] 对数缩放\n")
        try:
            method = int(input(""))
            if method not in [1, 2, 3]:
                print("请输入正确的数字！")
                continue
        except ValueError:
            print("请输入正确的数字！")
            continue
        break

    # 执行归一化
    for col in target_cols:
        if method == 1:
            x_df[col] = minmax_scale(x_df[col])
        elif method == 2:
            x_df[col] = zscore_scale(x_df[col])
        else:
            x_df[col] = log_scale(x_df[col])

    print(f"\n归一化完成！示例对比：")
    print(f"{'列名':<20} {'原始均值':>10} {'处理后均值':>10}")
    print("-" * 42)
    for col in target_cols:
        print(f"{col:<20} {original[col].mean():>10.4f} {x_df[col].mean():>10.4f}")
    return x_df

def apply_onehot_encoding(x_df):
    # 找出非数值列
    category_list = []
    for i in x_df.columns:
        if not pd.api.types.is_numeric_dtype(x_df[i]):
            category_list.append(i)
    
    if not category_list:
        print("无非数值列，跳过独热编码。")
        return x_df
    
    # 选择非数值列
    print("非数值列：")
    for idx, col in enumerate(category_list):
        print(f"{idx}, {col}")
    selected = input("选择需要独热编码的列(空格分割，直接回车=全部)：").split()
    if selected:
        target_cols = [category_list[int(i)] for i in selected]
    else:
        target_cols = category_list
    
    # 执行独热编码
    before = set(x_df.columns)
    x_df = pd.get_dummies(x_df, columns=target_cols)
    after = set(x_df.columns)
    new_cols = after - before
    print(f"独热编码完成！新增{len(new_cols)}列: {new_cols}")
    return x_df

def split_data(x_df, y_series, test_size=0.3, val_ratio=0.5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_series, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_ratio, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test