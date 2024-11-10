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
        # Multi-scale convolutional layers for different image scales
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # First scale
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),  # Second scale
            nn.Conv2d(3, 256, kernel_size=7, stride=1, padding=3),  # Third scale
        ])
        self.pool = nn.AdaptiveAvgPool2d((224, 224))  # To unify image sizes

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # Extract image features and apply multi-scale fusion
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feature = conv(pixel_values)
            pooled_feature = self.pool(feature)
            multi_scale_features.append(pooled_feature)

        # Combine the features from different scales
        combined_features = torch.cat(multi_scale_features, dim=1)

        # Use the default CLIP visual model for image encoding
        vision_outputs = self.visual.embeddings(combined_features)
        text_outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)

        # Continue with the original CLIP model logic
        logits_per_image = vision_outputs.logits_per_image
        logits_per_text = text_outputs.logits_per_text

        return logits_per_image, logits_per_text


class CrossModalAttention(nn.Module):
    def __init__(self, dim_image, dim_text):
        super(CrossModalAttention, self).__init__()
        # Multihead attention layer for combining image and text features
        self.attn = nn.MultiheadAttention(embed_dim=dim_image, num_heads=8)
        # A projection layer to map text features to the same dimension as image features
        self.text_projection = nn.Linear(dim_text, dim_image)  

    def forward(self, image_features, text_features):
        # Project text features to match the dimension of image features
        text_features = self.text_projection(text_features)

        # Concatenate the image and text features and perform attention
        features = torch.cat([image_features, text_features], dim=1)  # Concatenate along the feature dimension
        attn_output, _ = self.attn(features, features, features)
        return attn_output


class EnhancedCLIPModel(MultiScaleCLIPModel):
    def __init__(self, config):
        super(EnhancedCLIPModel, self).__init__(config)

        # Load the base CLIP model from a pre-trained checkpoint
        clip_model = CLIPModel.from_pretrained("./models")
        self.visual = clip_model.vision_model
        self.text_encoder = clip_model.text_model

        # Get the image and text encoding dimensions from the config
        image_dim = config.vision_config.hidden_size  # The size of the image feature vector
        text_dim = config.text_config.hidden_size  # The size of the text feature vector

        # Initialize CrossModalAttention with the respective dimensions
        self.cross_modal_attention = CrossModalAttention(dim_image=image_dim, dim_text=text_dim)

        # Add linear classifiers for both image and text features
        self.image_classifier = nn.Linear(image_dim, 1024)  # Assume 1024 categories for classification
        self.text_classifier = nn.Linear(text_dim, 1024)  # Classification for text

        # Projection layer for dimensionality reduction on text features
        self.text_projection = nn.Linear(1024, text_dim)  # Reduces dimensionality of text features

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # Get the outputs from the visual and text encoders
        vision_outputs = self.visual(pixel_values=pixel_values)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        image_features = vision_outputs.last_hidden_state
        text_features = text_outputs.last_hidden_state

        # Use CrossModalAttention to fuse image and text features
        fused_features = self.cross_modal_attention(image_features, text_features)

        # Pool image and text features separately
        pooled_image_features = torch.mean(fused_features[:, :image_features.size(1), :], dim=1)  # Image features pooling
        pooled_text_features = torch.mean(fused_features[:, image_features.size(1):, :], dim=1)  # Text features pooling

        # Reduce the dimension of text features using the projection layer
        pooled_text_features = self.text_projection(pooled_text_features)

        # Classify the pooled image and text features
        logits_per_image = self.image_classifier(pooled_image_features)
        logits_per_text = self.text_classifier(pooled_text_features)

        return logits_per_image, logits_per_text


class EnhancedTextEncoder(nn.Module):
    def __init__(self, config):
        super(EnhancedTextEncoder, self).__init__()
        # Initialize the text encoder from the pre-trained CLIP model
        self.text_encoder = CLIPModel.from_pretrained("./models").text
        self.layer_norm = nn.LayerNorm(config.hidden_size)  # Normalization layer
        self.dropout = nn.Dropout(0.1)  # Dropout layer for regularization
        self.additional_transformer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)

    def forward(self, input_ids, attention_mask):
        # Encode text using the CLIP text encoder
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_features = self.layer_norm(text_features)  # Apply normalization
        text_features = self.dropout(text_features)  # Apply dropout

        # Enhance the text features by passing them through an additional transformer layer
        enhanced_text_features = self.additional_transformer(text_features)

        return enhanced_text_features


class ImageTextDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 for model input
            transforms.ToTensor(),  # Convert images to tensor format
        ])
        # Load annotations from the JSON file
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Retrieve image and its corresponding description
        annotation = self.annotations[idx]
        image_path = os.path.join(self.images_dir, annotation["file_name"])
        description = annotation["description"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        # Preprocess image and text for CLIP
        inputs = self.processor(images=image, text=description, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        return inputs['pixel_values'][0], inputs['input_ids'][0], inputs['attention_mask'][0]

BATCH_SIZE = 100
LEARNING_RATE = 3e-5
NUM_EPOCHS = 20
WEIGHT_DECAY = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def collate_fn(batch):
    # Custom collate function to handle padding for input_ids and attention_masks
    images, input_ids, attention_masks = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    return images, input_ids, attention_masks


# Initialize the dataset and dataloader
images_dir = './images1107'
annotations_file = './shuffled_annotations5.json'
processor = CLIPProcessor.from_pretrained("./models")
dataset = ImageTextDataset(images_dir, annotations_file, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Load the model
model = EnhancedCLIPModel.from_pretrained("./models", ignore_mismatched_sizes=True)
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
num_training_steps = NUM_EPOCHS * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

# Log file to track training progress
log_file = "training_log(3.8w).txt"
with open(log_file, 'w') as f:
    f.write("Epoch, Batch, Loss\n")

# Training function for each epoch
def train_one_epoch(epoch):
    total_loss = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch') as pbar:
        for batch_idx, (images, input_ids, attention_mask) in
