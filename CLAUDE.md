# CLAUDE.md — ML Tool 项目上下文

## 项目背景

交互式机器学习工具箱，从 CLI 升级为 Web 应用（Flask）。目标：终端/浏览器里完成「加载数据 → 预处理 → 训练 → 评估」，为后续全栈 ML Playground 打基础。

这是**学习型项目**，用户在同步学习 ML 和 Web 开发。

## 教学方式（最重要）

用户明确要求：**「不要直接把代码写给我，让我自己写，你教我写」**。

- 只讲解思路、概念、API 用法，指出 bug，让用户自己动手写
- 写完后用户会让我检查，此时才读文件、指出问题
- 不要主动把完整代码贴给用户
- **回答要简短，问什么答什么**，不要长篇大段；用户没让看文件就不要主动读

## 技术栈

- Python 3.13.5（**虚拟环境 `.venv`，Python 路径 `.venv/Scripts/python.exe`**）
- Flask 3.1.3（服务端渲染 Jinja2）
- pandas、scikit-learn、PyTorch 2.12、NumPy
- sklearn `TfidfVectorizer`（文本特征）、`LabelEncoder`

## 项目结构

```
A:\PC_project\ML tool\
├── app.py            # Flask Web 版（主程序）
├── main.py           # CLI 版主菜单（已完成）
├── data_loader.py    # CLI：数据加载、浏览、统计
├── preprocess.py     # CLI：选列、归一化、独热、拆分
├── trainer.py        # CLI：PyTorch 训练
├── evaluator.py      # CLI：评估
├── templates/        # HTML 模板（index/browse/preprocess/normalize/train/evaluate/text_prepare 等）
├── uploads/          # 上传的数据集
├── 学习进度.md        # 学习进度记录
└── CLAUDE.md         # 本文件
```

## 当前进度

### Web 版已完成

| 页面/路由           | 功能                                                      |
| --------------- | ------------------------------------------------------- |
| `/`             | 上传 CSV，自动识别并删除行号列 + `Unnamed` 列；编码回退（UTF-8 失败转 latin-1） |
| `/browse`       | 数据预览 + 统计（数值列均值/标准差/最值，分类列唯一值）+ 缺失值计数                   |
| `/preprocess`   | 选标签列 + 特征列（数值流程）                                        |
| `/normalize`    | 归一化（3 方法，可选列）+ 独热编码（可选列）+ 缺失值处理（3 策略）+ 拆分               |
| `/train`        | 训练：分类/回归判断 + 用户可覆盖 + epochs/lr/**隐藏层大小** 可调             |
| `/evaluate`     | 评估：分类（准确率/精确率/召回率/F1/混淆矩阵），回归（MSE/RMSE/MAE/R²）          |
| `/text_prepare` | 文本情感分析：选文本列 + 标签列 → TF-IDF → 拆分 + 保存测试集 → 跳 `/train`（复用训练评估） |
| `/load`         | 加载历史模型：列出 `models/` 所有版本供选择 → 加载选中版本三件套 → 跳 `/evaluate` |
| `/predict`      | 文本预测：输入新文本 → vec 转 TF-IDF → 模型预测 → `inverse_transform` 还原标签（回填输入） |

所有路由已加前置保护，数据未就绪时返回友好提示。

### 关键决策

1. **架构**：Flask + Jinja2 服务端渲染，先跑通单体，之后再考虑前后端分离（Vue/React）
2. **分类/回归判断**（方案 C）：字符串标签强制分类；数值标签按唯一值 ≤10 判分类，用户可覆盖。逻辑在 `decide_type()`
3. **缺失值处理**：3 策略作用于整个 `x_df`，不选列
4. **归一化/独热编码**：用户可选列（checkbox）
5. **状态管理**：模块级全局变量（`df`、`x_df`、`y_series`、`X_train`...），无数据库
6. **特征统一为 numpy**：`X_train`/`X_val`/`X_test` 从拆分后统一转 numpy 数组（`.values`），`/train`、`/evaluate` 里不再写 `.values`。文本流程 TF-IDF 的 `.toarray()` 也产出 numpy，两边对齐
7. **文本流程复用训练评估**：`/text_prepare` 只做「文本→TF-IDF→拆分」并设置全局变量，然后跳 `/train`，训练评估完全复用（全局变量是天然接口）
8. **模型保存**：`model.pth` 存 dict（state_dict + input_dim + hidden1/2 + output_dim + is_classification + label_encoder），`/load` 据此重建架构
9. **多模型版本**（方案 B）：`/train` 训练完成后统一生成时间戳，三件套（`model_{ts}.pth` + `x_test_{ts}.csv` + `y_test_{ts}.csv`）存 `models/`，天然绑定；`/normalize`、`/text_prepare` 不再存 CSV。`/load` GET 列版本、POST 加载选中版本
10. **vec 保存**：`vec` 是全局变量，文本模型训练时 `joblib.dump` 存 `models/vec_{ts}.pkl`（数值模型不存）；`/load` 按文件存在与否读回并重置；`/predict` 用它把新文本转 TF-IDF

### 暂停/搁置

**实时 loss 曲线**（SSE + generator + Chart.js）—— 用户觉得前端 JS 太难，暂停，后续前端工作（含 CSS 美化）交给 Claude 做。

### 待办（未来方向）

1. **CSS 美化界面**（交给 Claude）
2. **实时 loss 曲线**（前端，交给 Claude）

## 技术细节与坑

### 环境

- 系统 Python 没装 Flask，测试必须用 `.venv/Scripts/python.exe`
- Windows GBK 编码：print 中文/emoji 会 `UnicodeEncodeError`，测试脚本加 `-X utf8`
- **改完 app.py 必须重启 Flask 才生效**（运行中的进程还是旧代码；`debug=True` 会自动重载则不用手动重启）

### Python

- 全局变量：**函数里赋值要 `global`，只读不用 `global`**
- 只要函数里出现 `df = ...`（赋值）就加 global；`df[...]` 读子集不用
- `request.form.get()` 用于 radio/select（单值），`request.form.getlist()` 用于 checkbox（多值）
- **`request.form` 对应 POST 表单，`request.args` 对应 URL 查询串**，别混
- `request.form.get("name", 默认值)` 第二参数是缺省值

### Flask / Jinja2

- `render_template("x.html", 左=右)`：左边是模板里用的名字，右边是 Python 变量
- **模板拿不到全局变量，必须通过 render_template 显式传入**
- `{{ 变量 }}` 输出，`{% if %} / {% for %}` 逻辑
- DataFrame 转表格：`df.to_html()`，模板里 `{{ table | safe }}`
- 路由 method 必须大写 `"GET"` / `"POST"`，`request.method` 也是大写；GET 和 POST 都要在 methods 里声明

### ML / 数据处理

- **LabelEncoder 跟着 `is_classification` 走，不是跟着 `is_numeric_dtype`**——字符串和数字标签分类都要编码（数字标签若非 0 开头如 1/2/3 也要编码，否则 CrossEntropyLoss 越界）
- 分类任务 `output_dim = len(类别数)`，回归 `output_dim = 1`
- **dtype 跟着「分类/回归」走**（分类 y=long，回归 y=float32），LabelEncoder 跟着「标签是否需转索引」走，两者独立
- 回归 y 转 tensor 要 `.reshape(-1, 1)`，分类用 `dtype=torch.long`
- `torch.max(output, 1)[1]` 拿预测类别索引（dim=1 沿列找每行最大）；分类要的是「最大值的索引」不是值本身
- 缺失值：`df.isnull().any().any()` 判断是否有 NaN；pandas `.mode()` 跳过 NaN，`statistics.mode()` 把 NaN 当众数（有坑）
- `df.dropna()` 是行操作，删除时 x 和 y 要同步（合并再删再拆）

### 文本 / TF-IDF

- `TfidfVectorizer` 三步：`fit`（学词表+IDF，只在训练集做一次）→ `transform`（转数字，训练/验证/测试都用同一个 vec）→ `.toarray()`（稀疏→稠密，torch 要稠密）
- **中文无空格**：用 `analyzer="char", ngram_range=(1,2)` 按字切 + 相邻双字，不用 jieba
- **`max_features` 限制列数**，否则词表巨大
- **稀疏矩阵只存非零格**：`(行, 列) 值` = `(第几文档, 第几词, 权重)`；`.toarray()` 才变稠密、才爆内存
- **稠密内存 ≈ 行数 × max_features × 4字节**；文本数据大就 `df.sample(10000)`（用 `sample` 不用 `head`，head 遇有序文件会取到有偏子集）
- 数据泄漏：TF-IDF 的 `fit` 必须在 `train_test_split` **之后**、只 fit 训练集
- **`transform`/`inverse_transform` 是批量 API**：单条也要包成列表 `transform([text])`、`inverse_transform([idx])`，返回数组取 `[0]`
- **`{% if pred_y %}` 有 0 坑**：预测结果 0/1 时 0 是假值不显示，判断「有没有结果」用 `{% if pred_y is not none %}`；表单回填：input 加 `value="{{ text }}"`

### 模型保存 / 加载

- 存 `state_dict`（权重），不存整个模型；加载前先**重建相同架构**再 `load_state_dict`
- 存 dict 打包重建所需的一切（state_dict + input_dim + hidden1/2 + output_dim + is_classification + label_encoder）
- **PyTorch 2.6+ `torch.load` 默认 `weights_only=True`**，文件里有 sklearn 对象（LabelEncoder）会报 `UnpicklingError`，要 `torch.load(path, weights_only=False)`
- **多模型版本化已解决错配问题**：三件套同时间戳绑定（见关键决策 #9）。旧坑记录：`model.pth` 与 `x_test.csv` 互不绑定时，换数据集会错配（测试集 4 列 vs 模型 5000 输入 → 矩阵乘法报错 `mat1 and mat2 shapes cannot be multiplied`）

### 训练 / 泛化

- **过拟合诊断**：train 准确率高但 val 明显低 = 过拟合；train 和 val 都 1.0 = 大概率数据问题（泄漏或数据有偏）
- 过拟合解法：减小模型、减 max_features、加 Dropout（`nn.Dropout(0.5)`，eval 时自动关闭）、更多数据

### 编码 / 数据集

- **CSV 编码不固定**：`pd.read_csv` 默认 UTF-8，英文数据集（如 spam.csv）常是 latin-1。读入时 `try utf-8 / except UnicodeDecodeError 转 latin-1`
- 文件末尾多余逗号会产生 `Unnamed: N` 垃圾列（少数行有值、多数 NaN），用 `df.drop(columns=[c for c in df.columns if "Unnamed" in c])` 删

### 数据集

- `iris.csv`：3 分类，`Unnamed: 0` 列**值从 1 开始**（不是 0），靠列名判断兜底
- `advertising.csv`：回归，预测 Sales
- `weibo_senti_100k.csv`：二分类情感，`label`(0/1) + `review`(文本)，已支持（TF-IDF char n-grams）
- `spam.csv`（UCI）：二分类，`v1`(ham/spam 字符串标签) + `v2`(短信文本)，latin-1 编码，尾部有 Unnamed 空列

## 测试方式

用 Flask test_client 做端到端测试：

```python
import app
from app import app as flask_app
flask_app.config['TESTING'] = True
client = flask_app.test_client()
```

测试时先手动设置 `app.df`、`app.X_train` 等全局变量模拟前置步骤，或用 `client.post('/preprocess', data={...})` 走完整流程。
