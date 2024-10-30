from tensorflow.keras import layers, models
import tensorflow as tf
from data_preprocessing import EarAcupointDataset

def unet_model(input_shape=(128, 128, 3), num_classes=1):
    inputs = layers.Input(input_shape)

    # 编码部分
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # 解码部分
    c2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(p1)
    concat = layers.concatenate([c1, c2])
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat)
    c2 = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c2)

    model = models.Model(inputs=[inputs], outputs=[c2])
    return model

def train_model(image_dir, annotation_file):
    # 读取标注数据
    annotations_df = pd.read_csv(annotation_file)
    dataset = EarAcupointDataset(image_dir, annotations_df)

    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    num_epochs = 10
    model.fit(dataset, epochs=num_epochs)

    return model
