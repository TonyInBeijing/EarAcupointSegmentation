from data_preprocessing import prepare_annotations
from model_training import train_model
import pandas as pd
import os


def main():
    # 数据准备
    json_dir = "data/annotations"  # JSON 文件所在目录
    prepare_annotations(json_dir)

    # 模型训练
    image_dir = "data/raw"  # 图像目录
    annotation_file = "data/annotations/annotations.csv"  # 标注文件
    model = train_model(image_dir, annotation_file)

    # 这里可以添加模型评估的代码
    # evaluate_model(model, some_validation_dataset)


if __name__ == "__main__":
    main()
