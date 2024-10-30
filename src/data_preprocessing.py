import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def extract_annotations(json_dir):
    data = []
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(json_dir, json_file)) as f:
                annotations = json.load(f)
                image_name = annotations['imagePath']
                
                for shape in annotations['shapes']:
                    label = shape['label']  # 获取穴位的标签
                    points = shape['points']  # 获取坐标点
                    for point in points:  # 如果每个穴位有多个点，可以遍历所有点
                        x, y = point
                        data.append((image_name, label, x, y))

    return pd.DataFrame(data, columns=['image_name', 'label', 'x', 'y'])

class EarAcupointDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, annotations_df, batch_size=16, image_size=(128, 128)):
        self.image_dir = image_dir
        self.annotations = annotations_df
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.annotations) / self.batch_size))

    def __getitem__(self, idx):
        batch_annotations = self.annotations.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        labels = []

        for _, row in batch_annotations.iterrows():
            img_path = os.path.join(self.image_dir, row['image_name'])
            image = load_img(img_path, target_size=self.image_size)
            image = img_to_array(image) / 255.0
            
            labels.append((row['x'], row['y']))  # 存储坐标信息

            images.append(image)

        return np.array(images), np.array(labels)

# 使用函数提取标注并保存为 CSV 文件
def prepare_annotations(json_dir):
    annotations_df = extract_annotations(json_dir)
    annotations_df.to_csv('data/annotations/annotations.csv', index=False)  # 保存为 CSV 文件
