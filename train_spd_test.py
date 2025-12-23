from ultralytics import YOLO

# from torchsummary import summary
# summary(model, (3, 640, 640))
model = YOLO("ultralytics/cfg/models/11/yolo11-spd.yaml")

print(">>> 开始训练测试...")
results = model.train(
    data="coco8.yaml",  # 使用内置的微型数据集，只有4张图训练，4张图验证
    epochs=10,  # 只跑3轮，确保存盘和打印日志没问题
    imgsz=640,  # 图片大小
    batch=2,  # 小一点，防止你电脑显存炸了
    project="runs/train",
    name="yolo11-spd-test",
    device="mps"  # mac为mps
)