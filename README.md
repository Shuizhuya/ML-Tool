# ML Tool — 交互式机器学习工具箱

基于 Flask 的交互式机器学习工具箱，在浏览器里完成「**加载数据 → 预处理 → 训练 → 评估 → 预测**」全流程。学习型项目，适合对 ML 和 Web 开发都感兴趣的人参考。

## 功能

- **上传数据**：CSV 上传，自动识别编码（UTF-8 / latin-1）、清理行号列和垃圾列
- **数据浏览**：前 10 行预览 + 数值列统计（均值/标准差/最值）+ 分类列唯一值 + 缺失值计数
- **数值流程**：选标签列/特征列 → 归一化（3 种方法）+ 独热编码 + 缺失值处理（3 种策略）→ 数据集拆分
- **文本流程**：TF-IDF 向量化（字符级 n-gram，支持中文无空格文本），自动随机采样防止内存爆炸
- **训练**：PyTorch 多层感知机，分类/回归自动判断（不确定类型可手动覆盖），epochs / 学习率 / 隐藏层大小可调，自动诊断过拟合
- **评估**：分类（准确率/精确率/召回率/F1/混淆矩阵），回归（MSE/RMSE/MAE/R²），预测错误样本展示
- **多模型版本管理**：训练自动保存「模型 + 测试集 + 向量器」三件套，支持命名、加载、删除
- **预测**：文本或数值输入，输出还原标签后的结果

## 快速开始

环境：Python 3.13+（Windows 11 已验证）

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
python -m venv .venv
.venv\Scripts\activate        # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器打开 http://127.0.0.1:5000

## 使用流程

**数值数据**：首页上传 CSV → 浏览数据 → 选列 → 数据预处理 → 训练 → 评估 → 预测

**文本数据**：首页上传 CSV → 文本预处理（选文本列 + 标签列）→ 训练 → 评估 → 预测

## 示例数据集

仓库不携带大数据集，可自行下载放入 `uploads/` 目录或通过首页直接上传：

| 数据集 | 任务 | 说明 |
| ------ | ---- | ---- |
| iris.csv | 3 分类 | [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) |
| advertising.csv | 回归 | 预测 Sales，Kaggle 搜索 advertising dataset |
| spam.csv | 二分类 | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)，latin-1 编码 |
| weibo_senti_100k.csv | 中文情感二分类 | 含 `label`(0/1) 和 `review`(文本) 两列 |

## 项目结构

```
├── app.py            # Flask Web 版主程序（核心）
├── main.py           # CLI 版主菜单
├── data_loader.py    # CLI：数据加载、浏览、统计
├── preprocess.py     # CLI：选列、归一化、独热、拆分
├── trainer.py        # CLI：PyTorch 训练
├── evaluator.py      # CLI：评估
├── templates/        # Jinja2 模板
├── static/           # CSS 等静态资源
├── uploads/          # 上传的数据集（运行时生成，不入库）
└── models/           # 模型版本三件套（运行时生成，不入库）
```

## 技术栈

Flask / Jinja2 / pandas / scikit-learn / PyTorch / NumPy / joblib

## 开发记录

- [CLAUDE.md](CLAUDE.md)：开发决策、踩坑记录、待办方向
- [学习进度.md](学习进度.md)：学习过程记录
