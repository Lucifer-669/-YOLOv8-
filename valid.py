from ultralytics import YOLO
import matplotlib.pyplot as plt

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run_model_validation():
    """
    执行模型验证流程
    使用验证集评估训练好的YOLOv8模型性能
    """
    # 模型和数据集配置
    MODEL_PATH = "xxx"  # 模型权重路径
    DATA_CONFIG = "xxx"  # 数据集配置路径

    # 验证参数设置
    BATCH_SIZE = 8  # 每批次处理图像数
    DEVICE = "0"  # 使用GPU设备
    IMAGE_SIZE = 640  # 输入图像尺寸
    RESULT_DIR = "val_report"  # 结果保存目录
    CONF_THRESH = 0.001  # 置信度阈值
    IOU_THRESH = 0.6  # IoU阈值

    # 加载训练好的模型
    trained_model = YOLO(MODEL_PATH)

    # 执行验证评估
    val_results = trained_model.val(
        data=DATA_CONFIG,
        batch=BATCH_SIZE,
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        name=RESULT_DIR,
        split="val",  # 使用验证集
        save_json=True,  # 保存JSON格式结果
        save_hybrid=True,  # 保存混合格式结果
        conf=CONF_THRESH,
        iou=IOU_THRESH
    )

    # 打印格式化评估报告
    print("\n================ 模型验证报告 ================")
    print(f"验证集 mAP@0.5:   {val_results.box.map50:.4f}")
    print(f"验证集 mAP@0.5-0.95: {val_results.box.map:.4f}")
    print(f"验证集 Precision: {val_results.box.p[0]:.4f}")
    print(f"验证集 Recall:    {val_results.box.r[0]:.4f}")
    print("============================================")


if __name__ == "__main__":
    # 执行验证流程
    run_model_validation()