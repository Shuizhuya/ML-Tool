df = None
X_train = X_val = X_test = None
y_train = y_val = y_test = None
model = None
import data_loader
import preprocess
import trainer
import evaluator
while True:
    print("=================================")
    print("          机器学习工具箱          ")
    print("=================================")
    print("[1] 加载数据集")
    print("[2] 浏览数据")
    print("[3] 数据预处理")
    print("[4] 训练模型")
    print("[5] 评估模型")
    print("[6] 退出")
    print("=================================")
    try:
        fun_num = int(input("请输入选项："))
    except ValueError:
        print("请输入数字！")
        continue
    if fun_num == 1:
        df = data_loader.load_dataset()
    elif fun_num == 2:
        if df is None or df.empty:
            print("请先加载数据集")
            continue
        data_loader.browse_data(df)
        ans = input("是否查看统计(y/n)？").strip().lower()
        if ans == "y":
            data_loader.show_stats(df)
    elif fun_num == 3:
        if df is None or df.empty:
            print("请先加载数据集")
            continue
        x_df, y_series, feature_cols, label_col = preprocess.select_columns(df)
        x_df = preprocess.apply_normalization(x_df)
        x_df = preprocess.apply_onehot_encoding(x_df)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess.split_data(x_df, y_series)
        print(f"训练集：{X_train.shape}， 验证集：{X_val.shape}， 测试集：{X_test.shape}")
    elif fun_num == 4:
        if X_train is None:
            print("请先完成数据预处理")
            continue
        model, label_encoder, is_classification = trainer.train_model(X_train, X_val, y_train, y_val)
    elif fun_num == 5:
        if model is None:
            print("请先训练模型")
            continue
        evaluator.evaluate_model(model, X_test, y_test, label_encoder, is_classification)
    elif fun_num == 6:
        break
        
