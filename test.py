import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 确保使用GPU进行预测，如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.cuda.is_available())
print(device)

# 加载训练好的模型和处理器
model_save_dir = "./run3.8w3e-510"  # 替换为您模型的保存路径
model = CLIPModel.from_pretrained(model_save_dir).to(device)
processor = CLIPProcessor.from_pretrained(model_save_dir)

# 定义预测函数
def predict(image_path):
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 处理输入
    inputs = processor(images=image, text=[
        "have person in four-wheeled truck container",
        "no person in four-wheeled truck container",
        "have person in three-wheeled truck container",
        "no person in three-wheeled truck container"
    ], return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        pixel_values=inputs['pixel_values'].to(device))

    logits = outputs.logits_per_image
    probabilities = logits.softmax(dim=1)  # 获取每个类别的概率
    predicted_class = torch.argmax(probabilities, dim=1)

    # 返回预测结果和概率
    return (
        [
            "have person in four-wheeled truck container",
            "no person in four-wheeled truck container",
            "have person in three-wheeled truck container",
            "no person in three-wheeled truck container"
        ][predicted_class.item()],
        probabilities[0].cpu().numpy()
    )

# 主函数
if __name__ == "__main__":
    # 指定包含图像的文件夹路径
    folder_path = "./ex_tricycle_legal_test"  # 预测文件夹路径

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在。")
    else:
        # 初始化类别计数
        category_counts = {
            "have person in four-wheeled truck container": 0,
            "no person in four-wheeled truck container": 0,
            "have person in three-wheeled truck container": 0,
            "no person in three-wheeled truck container": 0
        }

        # 遍历文件夹中的每个文件
        for filename in os.listdir(folder_path):
            # 检查文件扩展名
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)

                # 进行预测
                result, probabilities = predict(image_path)
                print(
                    f"文件: {filename}, 预测结果: {result}, 概率: "
                    f"有4: {probabilities[0]:.4f}, "
                    f"无4: {probabilities[1]:.4f}, "
                    f"有3: {probabilities[2]:.4f}, "
                    f"无3: {probabilities[3]:.4f}, "
                )

                # 更新类别计数
                category_counts[result] += 1

        # 打印每个类别的统计结果
        print("\n类别统计结果:")
        for category, count in category_counts.items():
            print(f"{category}: {count}")
