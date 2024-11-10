import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn


class MultiScaleCLIPModel(CLIPModel):
    def __init__(self, config):
        super(MultiScaleCLIPModel, self).__init__(config)
        # 多尺度卷积层
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 第一尺度
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),  # 第二尺度
            nn.Conv2d(3, 256, kernel_size=7, stride=1, padding=3),  # 第三尺度
        ])
        self.pool = nn.AdaptiveAvgPool2d((224, 224))  # 统一尺寸

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # 图像特征提取部分进行多尺度融合
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feature = conv(pixel_values)
            pooled_feature = self.pool(feature)
            multi_scale_features.append(pooled_feature)

        # 融合多尺度特征
        combined_features = torch.cat(multi_scale_features, dim=1)

        # 使用CLIP的默认图像编码器进行处理
        vision_outputs = self.visual.embeddings(combined_features)
        text_outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)

        # 继续使用原来的CLIP模型逻辑
        logits_per_image = vision_outputs.logits_per_image
        logits_per_text = text_outputs.logits_per_text

        return logits_per_image, logits_per_text


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8)

    def forward(self, image_features, text_features):
        # 将图像和文本特征拼接起来，然后进行注意力计算
        features = torch.cat([image_features, text_features], dim=1)  # 拼接
        attn_output, _ = self.attn(features, features, features)
        return attn_output

class CrossModalAttention(nn.Module):
    def __init__(self, dim_image, dim_text):
        super(CrossModalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_image, num_heads=8)
        self.text_projection = nn.Linear(dim_text, dim_image)  # 将文本特征维度转换为与图像相同的维度

    def forward(self, image_features, text_features):
        # 通过全连接层调整文本特征维度
        text_features = self.text_projection(text_features)

        # 将图像和文本特征拼接起来，然后进行注意力计算
        features = torch.cat([image_features, text_features], dim=1)  # 拼接
        attn_output, _ = self.attn(features, features, features)
        return attn_output


class EnhancedCLIPModel(MultiScaleCLIPModel):
    def __init__(self, config):
        super(EnhancedCLIPModel, self).__init__(config)

        clip_model = CLIPModel.from_pretrained("./models")
        self.visual = clip_model.vision_model
        self.text_encoder = clip_model.text_model

        # 获取 text_config 和 vision_config 中的 hidden_size
        image_dim = config.vision_config.hidden_size  # 可能需要调整这里
        text_dim = config.text_config.hidden_size  # 获取文本编码器的维度

        # 使用这两个维度初始化 CrossModalAttention
        self.cross_modal_attention = CrossModalAttention(dim_image=image_dim, dim_text=text_dim)

        # 添加线性层映射到类别空间
        self.image_classifier = nn.Linear(image_dim, 1024)  # 假设类别数为 1024
        self.text_classifier = nn.Linear(text_dim, 1024)  # 修改为 text_dim

        # 对文本特征进行降维
        self.text_projection = nn.Linear(1024, text_dim)  # 用于降维处理

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        vision_outputs = self.visual(pixel_values=pixel_values)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        image_features = vision_outputs.last_hidden_state
        text_features = text_outputs.last_hidden_state

        # 使用 CrossModalAttention 融合特征
        fused_features = self.cross_modal_attention(image_features, text_features)

        # 对于每个类别只提取一个特征 (pooling)
        pooled_image_features = torch.mean(fused_features[:, :image_features.size(1), :], dim=1)  # 图像特征
        pooled_text_features = torch.mean(fused_features[:, image_features.size(1):, :], dim=1)  # 文本特征

        # 降维文本特征
        pooled_text_features = self.text_projection(pooled_text_features)  # 映射到正确的维度

        # 映射到类别空间
        logits_per_image = self.image_classifier(pooled_image_features)  # 图像分类
        logits_per_text = self.text_classifier(pooled_text_features)  # 文本分类

        return logits_per_image, logits_per_text





class EnhancedTextEncoder(nn.Module):
    def __init__(self, config):
        super(EnhancedTextEncoder, self).__init__()
        self.text_encoder = CLIPModel.from_pretrained("./models").text
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.additional_transformer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)

    def forward(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_features = self.layer_norm(text_features)
        text_features = self.dropout(text_features)

        # 加入额外的Transformer层增强文本编码
        enhanced_text_features = self.additional_transformer(text_features)

        return enhanced_text_features

class ImageTextDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.images_dir, annotation["file_name"])
        description = annotation["description"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        inputs = self.processor(images=image, text=description, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        return inputs['pixel_values'][0], inputs['input_ids'][0], inputs['attention_mask'][0]

BATCH_SIZE = 19
LEARNING_RATE = 3e-5
NUM_EPOCHS = 20
WEIGHT_DECAY = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    return images, input_ids, attention_masks


# 数据加载器和训练集初始化
images_dir = './images1107'
annotations_file = './shuffled_annotations5.json'
processor = CLIPProcessor.from_pretrained("./models")
dataset = ImageTextDataset(images_dir, annotations_file, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 加载模型
model = EnhancedCLIPModel.from_pretrained("./models", ignore_mismatched_sizes=True)
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率调度器
num_training_steps = NUM_EPOCHS * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

# 日志记录
log_file = "training_log(3.8w).txt"
with open(log_file, 'w') as f:
    f.write("Epoch, Batch, Loss\n")

# 训练函数
def train_one_epoch(epoch):
    total_loss = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch') as pbar:
        for batch_idx, (images, input_ids, attention_mask) in enumerate(dataloader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            # 获取模型输出
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits_per_image = outputs[0]  # 形状: [batch_size, num_classes]
            logits_per_text = outputs[1]  # 形状: [batch_size, num_classes]

            # 创建目标标签，假设标签是连续的整数（从0到batch_size-1）
            labels = torch.arange(images.size(0)).to(device)  # labels的形状是 [batch_size]，即[19]

            # 计算损失
            loss_image = F.cross_entropy(logits_per_image, labels)  # logits_per_image: [batch_size, num_classes], labels: [batch_size]
            loss_text = F.cross_entropy(logits_per_text, labels)    # logits_per_text: [batch_size, num_classes], labels: [batch_size]

            # 总损失
            loss = (loss_image + loss_text) / 2

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # 写入日志
            with open(log_file, 'a') as f:
                f.write(f"{epoch + 1}, {batch_idx + 1}, {loss.item()}\n")

            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")


# 训练过程
for epoch in range(NUM_EPOCHS):
    train_one_epoch(epoch)

# 保存模型
model_save_dir = './run'
os.makedirs(model_save_dir, exist_ok=True)
model.save_pretrained(model_save_dir)
