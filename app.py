from flask import Flask, request, render_template, redirect, Response
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import joblib
app = Flask(__name__)
df = None
y_series = None
x_df = None
X_train = X_val = X_test = y_train = y_val = y_test = None
label_encoder = is_classification = model = None
vec = raw_X_test = None
feature_names = None
def decide_type(y_train, y_val):
    if not pd.api.types.is_numeric_dtype(y_train):
        return "classification"
    list1 = y_train.unique()
    list2 = y_val.unique()
    num_list = list(set(list1) | set(list2))
    if len(num_list) <= 10:
        return "classification"
    return "regression"

@app.route("/", methods=["GET", "POST"])
def home():
    global df
    if request.method == "POST":
        file = request.files["dataset"]
        if file:
            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", file.filename)
            file.save(path)
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
            
            first_col = df.columns[0]
            col_values = df[first_col].values
            reference = range(len(df))
            if list(col_values) == list(reference) or df.columns[0] == "Unnamed: 0":
                df = df.drop(columns=first_col) # 删除索引行
            df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
            return redirect("/browse")
    return render_template("index.html")

@app.route("/browse")
def browse():
    global df
    if df is None:
        return render_template("message.html", message = "请先上传数据集")
    table_html = df.head(10).to_html()
    stats = []
    for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats.append({
                    "name" : col,
                    "type" : "数值",
                    "mean" : round(df[col].mean(), 2),
                    "std" : round(df[col].std(), 2),
                    "min" : round(df[col].min(), 2),
                    "max" : round(df[col].max(), 2),
                    "missing" : int(df[col].isnull().sum())
                })
            else:
                stats.append({
                    "name" : col,
                    "type" : "分类",
                    "unique" : df[col].nunique(),
                    "missing" : int(df[col].isnull().sum())
                 })
    return render_template("browse.html", table = table_html, stats = stats)

@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    global df, x_df, y_series
    if df is None:
        return render_template("message.html", message = "请先上传数据集")
    if request.method == "POST":
        label_col = request.form.get("label_col")
        feature_cols = request.form.getlist("feature_cols")
        y_series = df[label_col]
        x_df = df[feature_cols]
        return redirect("/normalize")
    columns = df.columns.tolist()
    return render_template("preprocess.html", columns = columns)

@app.route("/normalize", methods=["GET", "POST"])
def normalize():
    global x_df, y_series,  X_train, X_val, X_test, y_train, y_val, y_test, feature_names, vec, raw_X_test
    if x_df is None:
        return render_template("message.html", message = "请先完成数据预处理（选列）")
    if request.method == "POST":
        vec = None
        raw_X_test = None
        method = int(request.form.get("method"))
        norm_cols = request.form.getlist("norm_cols")
        cate_cols = request.form.getlist("cate_cols")
        missing_way = request.form.get("missing_way")
        if x_df.isnull().any().any():
            if missing_way == "drop":
                combined = pd.concat([x_df, y_series], axis=1)
                combined = combined.dropna()
                y_series = combined.iloc[:, -1]
                x_df = combined.iloc[:, :-1]
            elif missing_way == "mean":
                for col in x_df.select_dtypes(include="number").columns:
                    x_df[col] = x_df[col].fillna(x_df[col].mean())
                for col in x_df.select_dtypes(exclude="number").columns:
                    x_df[col] = x_df[col].fillna(x_df[col].mode()[0])
            else:
                for col in x_df.select_dtypes(include="number").columns:
                    x_df[col] = x_df[col].fillna(x_df[col].median())
                for col in x_df.select_dtypes(exclude="number").columns:
                    x_df[col] = x_df[col].fillna(x_df[col].mode()[0])
        for i in norm_cols:
            if method == 1:
                x_df[i] = (x_df[i] - x_df[i].min()) / (x_df[i].max() - x_df[i].min())
            elif method == 2:
                x_df[i] = (x_df[i] - x_df[i].mean()) / x_df[i].std()
            else:
                x_df[i] = np.log(x_df[i])
        if cate_cols:
            x_df = pd.get_dummies(x_df, columns=cate_cols)
        X_train, X_test, y_train, y_test = train_test_split(x_df, y_series, test_size=0.3, random_state=42)
        feature_names = list(X_train.columns)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        X_train, X_test, X_val = X_train.values, X_test.values, X_val.values
        return render_template("normalize.html", train_shape = f"{X_train.shape[0]}行, {X_train.shape[1]}列", test_shape = f"{X_test.shape[0]}行, {X_test.shape[1]}列", val_shape = f"{X_val.shape[0]}行, {X_val.shape[1]}列")
    numeric_cols = [col for col in x_df.columns if pd.api.types.is_numeric_dtype(x_df[col])]
    category_cols = [col for col in x_df.columns if not pd.api.types.is_numeric_dtype(x_df[col])]
    return render_template("normalize_form.html", numeric_cols = numeric_cols, category_cols = category_cols)

@app.route("/text_prepare", methods=["GET", "POST"])
def text_prepare():
    global df, x_df, X_train, X_test, X_val, y_series, y_train, y_test, y_val, vec, raw_X_test
    if df is None:
        return render_template("message.html", message = "请先加载数据集")
    if request.method == "GET":
        columns = df.columns.tolist()
        return render_template("text_prepare.html", columns = columns)
    else:
        if len(df) > 10000:
            df = df.sample(10000, random_state = 42)
        x_df = df[request.form.get("x_df")]
        y_series = df[request.form.get("y_series")]
        X_train, X_test, y_train, y_test = train_test_split(x_df, y_series, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        raw_X_test = X_test
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=5000)
        vec.fit(X_train)
        X_train = vec.transform(X_train).toarray().astype(np.float32)
        X_test = vec.transform(X_test).toarray().astype(np.float32)
        X_val = vec.transform(X_val).toarray().astype(np.float32)
        return redirect("/train")


@app.route("/train", methods=["GET", "POST"])
def train():
    global label_encoder, is_classification, model
    if request.method == "POST":
        if X_train is None:
            return render_template("message.html", message = "请先完成数据预处理")
        task_type = request.form.get("task_type")
        if task_type is None:
            task_type = "classification"
        is_classification = (task_type == "classification")
        hidden1 = int(request.form.get("hidden1", 64))
        hidden2 = int(request.form.get("hidden2", 32))
        label_encoder = None
        if is_classification:
            label_encoder = LabelEncoder()
            y_train_encoder = label_encoder.fit_transform(y_train)
            y_val_encoder = label_encoder.transform(y_val)
            list1 = y_train.unique()
            list2 = y_val.unique()
            num_list = list(set(list1) | set(list2))
            output_dim = len(num_list)
            y_train_t = torch.tensor(y_train_encoder, dtype=torch.long)
            y_val_t = torch.tensor(y_val_encoder, dtype=torch.long)
        else:
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
            y_val_t = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
            output_dim = 1
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        input_dim = X_train_t.size(1)
        model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )
        if is_classification:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()

        history = []
        epochs = int(request.form.get("epochs", 50))
        lr = float(request.form.get("lr", 0.001))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 9:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = loss_fn(val_pred, y_val_t)
                history.append({"epoch": epoch, "train_loss": round(loss.item(), 5), "val_loss": round(val_loss.item(), 5)})
        message = None
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t)
            val_pred = model(X_val_t)
            train_acc = val_acc = None
            train_loss = val_loss = None
            if is_classification:
                train_idx = torch.max(train_pred, 1)[1]
                train_y = train_idx.numpy()
                val_idx = torch.max(val_pred, 1)[1]
                val_y = val_idx.numpy()
                train_acc = accuracy_score(y_train_t, train_y)
                val_acc = accuracy_score(y_val_t, val_y)
                if train_acc - val_acc > 0.15:
                    message = "模型疑似过拟合"
                if train_acc == 1.0 or val_acc == 1.0:
                    message = "训练集或验证集准确率为1.0, 可能出现数据泄露或有偏"
            else:
                train_y = train_pred.numpy().flatten()
                val_y = val_pred.numpy().flatten()
                train_loss = loss_fn(train_y, y_train_t)
                val_loss = loss_fn(val_y, y_val_t)
                if val_loss / train_loss >= 2:
                    message = "模型疑似过拟合"
        os.makedirs("models", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({"state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "output_dim": output_dim,
                    "is_classification": is_classification,
                    "label_encoder": label_encoder,
                    "feature_names": feature_names}, f"models/model_{ts}.pth")
        pd.DataFrame(X_test).to_csv(f"models/x_test_{ts}.csv", index=False)
        y_test.to_csv(f"models/y_test_{ts}.csv", index=False)
        if vec is not None:
            joblib.dump(vec, f"models/vec_{ts}.pkl")
            pd.DataFrame(raw_X_test).to_csv(f"models/x_raw_test_{ts}.csv", index=False)
        return render_template("train.html", history = history, task_type = task_type, numeric_label = pd.api.types.is_numeric_dtype(y_train), ts = ts, message = message, train_acc = train_acc, val_acc = val_acc, train_loss = train_loss, val_loss = val_loss)
    else:
        if X_train is None:
            return render_template("message.html", message = "请先完成数据预处理")
        task_type = decide_type(y_train, y_val)
        numeric_label = pd.api.types.is_numeric_dtype(y_train)
        return render_template("train.html", task_type = task_type, numeric_label = numeric_label)

@app.route("/name", methods=["POST"])
def name():
    ts = request.form.get("ts")
    if ts is None:
        return render_template("message.html", message = "版本信息缺失, 请训练后从结果页命名")
    name = request.form.get("name")
    if not name or not name.strip():
        return render_template("message.html", message = "名字不能为空")
    if os.path.exists("models/names.json") and os.path.getsize("models/names.json") > 0: # 文件存在且内容不为空时读取
        with open("models/names.json", "r", encoding="utf-8") as f1:
            names = json.load(f1)
            names[ts] = name
    else:
        names = {}
        names[ts] = name
    with open("models/names.json", "w", encoding="utf-8") as f2:
        json.dump(names, f2, ensure_ascii=False)
    return redirect("/load")

@app.route("/load", methods=["GET", "POST"])
def load():
    global model, is_classification, label_encoder, X_test, y_test, vec, feature_names, raw_X_test
    if request.method == "GET":
        if not os.path.exists("models"):
            return render_template("message.html", message = "还未训练过，请先进行训练")
        files = os.listdir("models")
        ts = []
        for f in files:
            if f.endswith(".pth"):
                ts.append(f.replace("model_", "").replace(".pth", ""))
        if os.path.exists("models/names.json") and os.path.getsize("models/names.json") > 0: # 文件存在且内容不为空时读取
                with open("models/names.json", "r", encoding="utf-8") as f1:
                    names = json.load(f1)
        else:
            names = {}
        return render_template("load.html", ts = ts, names = names)
    else:
        ts = request.form.get("ts")
        if ts is None:
            return render_template("message.html", message = "请选择要加载的模型版本")
        if os.path.exists(f"models/vec_{ts}.pkl"):
            vec = joblib.load(f"models/vec_{ts}.pkl")
        else:
            vec = None
        if os.path.exists(f"models/x_raw_test_{ts}.csv"):
            raw_X_test = pd.read_csv(f"models/x_raw_test_{ts}.csv").iloc[:, 0].values
        else:
            raw_X_test = None
        reuse = torch.load(f"models/model_{ts}.pth", weights_only=False)
        X_test = pd.read_csv(f"models/x_test_{ts}.csv").values
        y_test = pd.read_csv(f"models/y_test_{ts}.csv").iloc[:, 0].values # read_csv读出来为dataframe,用iloc转为series
        is_classification = reuse['is_classification']
        label_encoder = reuse['label_encoder']
        feature_names = reuse['feature_names']
        hidden1 = reuse['hidden1']
        hidden2 = reuse['hidden2']
        model = nn.Sequential(
                nn.Linear(reuse['input_dim'], hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, reuse['output_dim'])
            )
        model.load_state_dict(reuse['state_dict'])
        model.eval()
    return redirect("/evaluate")

@app.route("/delete", methods=["POST"])
def delete():
    ts = request.form.get("ts")
    if ts is None:
        return render_template("message.html", message = "请选择要删除的版本")
    os.remove(f"models/model_{ts}.pth")
    os.remove(f"models/x_test_{ts}.csv")
    os.remove(f"models/y_test_{ts}.csv")
    if os.path.exists(f"models/vec_{ts}.pkl"):
        os.remove(f"models/vec_{ts}.pkl")
    if os.path.exists("models/names.json") and os.path.getsize("models/names.json"):
        with open("models/names.json", "r", encoding="utf-8") as f3:
            names = json.load(f3)
        if ts in names:
            del names[ts]
        with open("models/names.json", "w", encoding="utf-8") as f4:
            json.dump(names, f4, ensure_ascii=False)
    if os.path.exists(f"models/x_raw_test_{ts}.csv"):
        os.remove(f"models/x_raw_test_{ts}.csv")
    return redirect("/load")

@app.route("/evaluate")
def evaluate():
    if model is None:
        return render_template("message.html", message = "请先训练模型")
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(X_test_t)
        if is_classification:
            indexs = torch.max(output, 1)[1]
            y_pred = indexs.numpy()
        else:
            y_pred = output.numpy().flatten()

    if label_encoder:
        y_true = label_encoder.transform(y_test)
    else:
        y_true = y_test

    mask = (y_pred != y_true)
    error_count = mask.sum()
    x_miss = X_test[mask]
    if raw_X_test:
        x_miss = raw_X_test[mask]
    if is_classification:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        confusion = confusion_matrix(y_true, y_pred)
        true_y = label_encoder.inverse_transform(y_true[mask])
        miss_y = label_encoder.inverse_transform(y_pred[mask])
        errors = list(zip(x_miss, miss_y, true_y))
        return render_template("evaluate.html", is_classification = is_classification, accuracy = accuracy, precision = precision, recall = recall, f1 = f1, confusion = confusion, errors = errors, error_count = error_count)
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        errors = list(zip(x_miss, y_pred[mask], y_true[mask]))
        return render_template("evaluate.html", is_classification = is_classification, mse = mse, rmse = rmse, mae = mae, r2 = r2, errors = errors, error_count = error_count)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        if model is None:
            return render_template("message.html", message = "请先加载模型")
        if vec is not None:
            text = request.form.get("text")
            x = vec.transform([text]).toarray().astype(np.float32)
            x_t = torch.tensor(x)
            model.eval()
            with torch.no_grad():
                output = model(x_t)
                if is_classification:
                    idx = torch.max(output, 1)[1].item()
                    pred_y = label_encoder.inverse_transform([idx])[0] # [idx]长度为1, 返回1个标签的数组, 所以取[0]
                else:
                    pred_y = output.item()
            return render_template("predict.html", pred_y = pred_y, text = text)
        else:
            vals = []
            for i in range(1, len(feature_names) + 1):
                vals.append(float(request.form.get(f"f{i}")))
            x = np.array([vals], dtype=np.float32)
            x_t = torch.tensor(x, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                output = model(x_t)
                if is_classification:
                    idx = torch.max(output, 1)[1].item()
                    pred_y = label_encoder.inverse_transform([idx])[0]
                else:
                    pred_y = output.item()
            return render_template("predict.html", pred_y = pred_y)
    if model is None:
        return render_template("message.html", message = "请先加载模型")
    return render_template("predict.html", is_text = (vec is not None), feature_names = feature_names, pred_y = None)
if __name__ == "__main__":
    app.run(debug=True)