from ultralytics import YOLO

# 1. 实例化你的魔改模型
# 注意：这里加载的是 .yaml 文件，表示从头构建网络结构
model = YOLO("ultralytics/cfg/models/11/yolo11-spd.yaml")

# 2. 加载预训练权重 (可选，但推荐)
# 虽然我们要从头训练，但加载部分权重可以加速收敛。
# 系统会自动忽略掉层数不匹配的部分（也就是你改掉的前两层）。
try:
    model.load("yolo11n.pt")
    print(">>> 成功加载通用权重，未匹配层将被随机初始化 (正常现象)")
except Exception as e:
    print(f">>> 权重加载跳过或部分加载: {e}")

# 3. 开始“冒烟测试” (Smoke Test)
# 我们只跑 3 个 epoch，用极小的数据集 coco8
print(">>> 开始训练测试...")
results = model.train(
    data="coco8.yaml",  # 使用内置的微型数据集，只有4张图训练，4张图验证
    epochs=3,  # 只跑3轮，确保存盘和打印日志没问题
    imgsz=640,  # 图片大小
    batch=2,  # 小一点，防止你电脑显存炸了
    project="runs/train",
    name="yolo11-spd-test",
    device="mps"  # mac为mps
)

print(">>> 测试结束！如果没有报错，恭喜你，SPD-Conv 植入成功！")
