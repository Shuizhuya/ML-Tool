import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
def evaluate_model(model, X_test, y_test, label_encoder, is_classification):
    x_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    # 得出预测值
    model.eval()
    with torch.no_grad():
        output = model(x_test_t)
        if is_classification:
            indexs = torch.max(output, 1)[1]
            y_pred = indexs.numpy() # 转换类型
        else:
            y_pred = output.numpy().flatten()

    if label_encoder:
        y_true = label_encoder.transform(y_test)
    else:
        y_true = y_test.values

    # 对比结果，求指标
    if is_classification:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        confusion = confusion_matrix(y_true, y_pred)
        print(f"准确率：{accuracy}，精确率：{precision}，召回率：{recall}，F1得分：{f1}，混淆矩阵：{confusion}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": confusion}
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"MSE:{mse},RMSE:{rmse},MAE:{mae},R^2:{r2}")
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
