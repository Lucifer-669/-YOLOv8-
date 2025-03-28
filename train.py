from ultralytics import YOLO


def train_fall_detection_model():

    # 模型配置
    MODEL_ARCH = "yolov8s.yaml"  # 模型结构文件
    DATA_CONFIG = "dataset/data.yaml"  # 数据集配置文件路径

    # 训练超参数
    TRAINING_EPOCHS = 100  # 训练轮次
    IMAGE_SIZE = 640  # 输入图像尺寸
    BATCH_SIZE = 8  # 批量大小
    DEVICE = "0"  # 使用GPU设备
    INITIAL_LR = 0.01  # 初始学习率
    EARLY_STOP = 15  # 早停耐心值
    SAVE_INTERVAL = 10  # 模型保存间隔

    # 数据增强配置
    USE_AUGMENTATION = True  # 是否启用数据增强
    FLIP_LR_PROB = 0.2  # 水平翻转概率
    FLIP_UD_PROB = 0.1  # 垂直翻转概率
    ROTATE_DEGREES = 10  # 旋转角度范围
    USE_MOSAIC = 0.0  # 禁用Mosaic增强

    # 初始化模型
    model = YOLO(MODEL_ARCH)

    # 执行训练
    model.train(
        data=DATA_CONFIG,
        epochs=TRAINING_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        augment=USE_AUGMENTATION,
        lr0=INITIAL_LR,
        patience=EARLY_STOP,
        save_period=SAVE_INTERVAL,
        name="fall_detection_v1",
        exist_ok=True,
        cos_lr=True,
        fliplr=FLIP_LR_PROB,
        flipud=FLIP_UD_PROB,
        degrees=ROTATE_DEGREES,
        mosaic=USE_MOSAIC,
    )


if __name__ == "__main__":
    train_fall_detection_model()