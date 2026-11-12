import pandas as pd

def load_dataset():
    path = input("请输入文件路径：")
    has_header = input("文件是否有表头(y/n)?:").strip().lower()
    kwargs = {}
    if has_header == 'n':
        kwargs['header'] = None
    try:
        df = pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        print("文件路径不存在/错误！")
        return None
    except Exception as e:
        print(f"文件读取出错：{e}")
        return None
    if has_header == 'n':
        while True:
            new_names = input("请分别输入列名(以空格分隔)：").split()
            if len(df.columns) == len(new_names):
                df.columns = new_names
                break
            else:
                print(f"输入列名数量不匹配，需要{len(df.columns)}个，实际{len(df.columns)}个，请重新输入。")
    return df

def browse_data(df):
    print("前十行数据为：\n")
    print(df.head(10))
    return

def show_stats(df):
    # print(df.describe())
    # return
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # 数值列
            print(f"均值：{df[col].mean():.2f}, 标准差：{df[col].std():.2f}")
            print(f"最小值：{df[col].min():.2f}, 最大值：{df[col].max():.2f}")
        else:
            # 分类列
            print(f"唯一值数量：{df[col].nunique()}")
            print(f"各类数量：\n{df[col].value_counts()}")
    return