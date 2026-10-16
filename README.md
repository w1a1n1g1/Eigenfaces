# Eigenfaces 人脸识别：传统方法 vs 轻量 CNN

基于 ORL 人脸数据集（40 人 × 10 张），实现并对比两类人脸识别方案：

- **传统方案**：LBP 纹理特征 + PCA 降维（Eigenfaces）+ KNN 分类
- **深度学习方案**：搭建轻量化 CNN，端到端完成特征提取与分类

项目包含完整的预处理（灰度化、尺寸归一化、训练/测试集划分）、训练与评估流程，并输出准确率柱状图、CNN 训练曲线与运行耗时对比，便于直观理解两类方法的差异。

## 功能特性

- **数据预处理**：自动下载/解压 ORL 数据集，完成图像读取、灰度转换、尺寸归一化与数据划分
- **传统识别流程**：LBP 纹理特征提取、PCA 特征脸降维、KNN 分类，含模型训练、测试与准确率评估
- **深度学习流程**：轻量化 CNN（卷积-池化特征提取 + 全连接分类），完成模型训练与性能评估
- **结果可视化**：两种方法准确率柱状图对比、CNN 训练曲线、运行耗时统计
- **Streamlit 演示应用**：上传人脸图片即可完成识别，展示「输入 → 匹配结果」

## 快速开始

```bash
# 1. 安装依赖
pip install numpy opencv-python scikit-learn tensorflow matplotlib streamlit

# 2. 准备数据集（仓库已附带 att_faces.zip，也可用脚本重新下载）
python download_dataset.py
python prepare_dataset.py

# 3. 运行传统方案（LBP + PCA + KNN）
python main.py

# 4. 启动可视化演示
python start_app.py   # 或 streamlit run app.py
```

## 项目结构

```
Eigenfaces/
├── app.py                  # Streamlit 可视化应用
├── start_app.py            # 应用启动脚本
├── main.py                 # 传统方案主流程（LBP + PCA + KNN）
├── create_dataset.py       # 自建人脸数据采集
├── download_dataset.py     # ORL 数据集下载脚本
├── prepare_dataset.py      # 数据预处理与划分
├── create_test_image.py    # 测试图像生成
├── att_faces.zip           # ORL 人脸数据集（400 张，40 人）
├── face_dataset/           # 解压后的人脸数据
├── models/                 # 模型权重与结果输出
├── test_face.jpg           # 测试图像示例
└── .streamlit/             # Streamlit 配置
```

## 方案对比要点

| 维度 | 传统方案（LBP+PCA+KNN） | 轻量 CNN |
|------|--------------------------|----------|
| 特征 | 手工纹理特征 + 特征脸投影 | 卷积自动学习特征 |
| 训练成本 | 极低，无需 GPU | 需要训练轮次与算力 |
| 可解释性 | 特征脸可视化直观 | 依赖训练曲线分析 |
| 适用场景 | 小样本、资源受限 | 精度上限更高 |

> 具体准确率与耗时数据以仓库运行输出的柱状图与训练曲线为准。

## 局限性

- ORL 为小规模数据集（400 张），结论在更大规模、复杂光照/姿态场景下未必成立
- CNN 训练结果受随机种子与超参数影响，图表为单次运行结果

## License

MIT License
