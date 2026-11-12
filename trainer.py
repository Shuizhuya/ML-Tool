import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

def train_model(X_train, X_val, y_train, y_val):
    # 判断是分类或回归
    list1 = y_train.unique()
    list2 = y_val.unique()
    num_list = list(set(list1) | set(list2))
    is_classification = False
    label_encoder = None
    if len(num_list) <= 10:
        is_classification = True
        print(f"检测为分类任务{len(num_list)}个类别")
        label_encoder = LabelEncoder()
        y_train_encoder = label_encoder.fit_transform(y_train)
        y_val_encoder = label_encoder.transform(y_val) # 使用同一映射，不用再fit
    else:
        print("检测为回归任务")
    
    # 将dataframe转变为tensor
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    if is_classification:
        y_train_t = torch.tensor(y_train_encoder, dtype=torch.long)
        y_val_t = torch.tensor(y_val_encoder, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)


    # 定义模型  
    output_dim = len(num_list) if is_classification else 1
    model = nn.Sequential(
        nn.Linear(X_train_t.size(1), 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )
    
    # 训练循环
    optimizer = torch.optim.Adam(model.parameters())
    if is_classification:
        # 分类模型的损失函数
        loss_fn = nn.CrossEntropyLoss()
    else:
        # 回归模型的损失函数
        loss_fn = nn.MSELoss()

    epochs = 50
    for epoch in range(epochs):
        # 训练模式
        model.train()
        # 清空前梯度
        optimizer.zero_grad()
        # 正向传播
        pred = model(X_train_t)
        # 计算损失
        loss = loss_fn(pred, y_train_t)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    
        if epoch % 10 == 9:
            # 评估模式
            model.eval()
            # 不记录更新梯度
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t)
            print(f"Epoch{epoch}: train_loss={loss.item():.5f}, val_loss={val_loss.item():.5f}")
    
    print("训练完成！")
    return model, label_encoder, is_classification