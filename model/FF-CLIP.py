'''
This is pytorch impplementation of FF-CLIP from
Can Language Improve Visual Features For Distinguishing Unseen Plant Diseases?
https://link.springer.com/chapter/10.1007/978-3-031-78113-1_20

'''

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import os
import cv2
from albumentations import Compose, Normalize, Resize, CenterCrop
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from collections import OrderedDict
from clip.model import Transformer, ModifiedResNet
from typing import Tuple, Union
from timm.layers import trunc_normal_
from torchvision.transforms import ToTensor
import csv
import random
import torch.distributed as dist

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def dtype(visual):
    return visual.conv1.weight.dtype
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class CustomDatasetForTestFFCLIP(Dataset):
    def __init__(self, csv_path, transforms, data_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels for plant
        self.label_arr_p = np.asarray(self.data_info.iloc[:, 1])
        # Thrid column is the labels for disease
        self.label_arr_d = np.asarray(self.data_info.iloc[:, 2])
        # Fourth column is the text for plant
        self.text_arr_p = np.asarray(self.data_info.iloc[:, 3])
        # Fifth column is the text for disease
        self.text_arr_d = np.asarray(self.data_info.iloc[:, 4])

        self.title_p = clip.tokenize(self.text_arr_p)
        self.title_d = clip.tokenize(self.text_arr_d)
        
        self.transforms = transforms
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.data_path = data_path

    def __getitem__(self, index):

        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.data_path, single_image_name)
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # Transform image to tensor
        transformed = self.transforms(image=image)
        image = transformed["image"]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label_p = self.label_arr_p[index]
        single_image_label_d = self.label_arr_d[index]
        single_text_label_p = self.title_p[index]
        single_text_label_d = self.title_d[index]
        return (image, single_image_label_p, single_text_label_p, single_image_label_d, single_text_label_d)        
    
    def __len__(self):
        return self.data_len


class CustomDatasetForTrainFFCLIP(Dataset):
    def __init__(self, plant_csv_dir, disease_csv_dir, plant_image_dir, disease_image_dir, transform=None):
        """
        Args:
            plant_csv_dir (str): Path to the plant CSV folder.
            disease_csv_dir (str): Path to the disease CSV folder.
            plant_image_dir (str): Path to the folder containing plant images.
            disease_image_dir (str): Path to the folder containing disease images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transforms = transform
        self.epoch = 0
        self.rank = 0
        self.world_size = 1

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Directories where the images are stored
        self.plant_image_dir = plant_image_dir
        self.disease_image_dir = disease_image_dir

        # Load plant and disease data
        self.plant_data = self._load_data(plant_csv_dir, label_idx=1, text_idx=2)
        self.disease_data = self._load_data(disease_csv_dir, label_idx=1, text_idx=2)
        self.epoch_pairs = []

        self.refresh_pairs()

    def _load_data(self, folder, label_idx, text_idx):
        """Loads CSV and returns list of (image_name, label, text)."""
        data = []
        for csv_file in sorted(os.listdir(folder)):
            if not csv_file.endswith('.csv'):
                continue
            csv_path = os.path.join(folder, csv_file)
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) > max(label_idx, text_idx):
                        image_name = row[0]
                        label = int(row[label_idx])
                        text = row[text_idx]
                        data.append((image_name, label, text))
        return data

    def refresh_pairs(self):
        """Shuffle disease data and pair with plant data 1:1 (with wrapping)."""
        rng = random.Random(self.epoch + self.rank * 10000)

        shuffled_disease = self.disease_data.copy()
        rng.shuffle(shuffled_disease)

        self.epoch_pairs = [
            (plant, shuffled_disease[i % len(shuffled_disease)])
            for i, plant in enumerate(self.plant_data)
        ]

    def set_epoch(self, epoch):
        """Call this at the start of each epoch (especially in DDP)."""
        self.epoch = epoch
        self.refresh_pairs()

    def __len__(self):
        return len(self.epoch_pairs)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Get the image names and labels
        (plant_image_name, plant_label, plant_text), (disease_image_name, disease_label, disease_text) = self.epoch_pairs[idx]

        # Build the full image paths
        plant_image_path = os.path.join(self.plant_image_dir, plant_image_name)
        disease_image_path = os.path.join(self.disease_image_dir, disease_image_name)

        image = cv2.imread(plant_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=image)
        transformed1 = transformed["image"]        
        
        image = cv2.imread(disease_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Transform image to tensor
        transformed = self.transforms(image=image)
        transformed2 = transformed["image"]        
      
        plant_text = clip.tokenize(plant_text).squeeze(0)
        disease_text = clip.tokenize(disease_text).squeeze(0)

        return transformed1, plant_label, plant_text, transformed2, disease_label, disease_text
    
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward_visual_features(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_image_features(self, image):
        return self.visual.forward_visual_features(image.type(self.dtype))
    
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_features(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]

        # return logits_per_image, logits_per_text
        
        # From Abel        
        return image_features, text_features, logits_per_image, logits_per_text

    # def forward_abel_image_clip(self, image):
    def forward_image_clip_features(self, image):
        image_features = self.encode_image_features(image)

        return image_features

    def forward_text_clip_features(self, text):
        text_features = self.encode_text_features(text)
        return text_features

    
class modelclassifierwithclip(nn.Module):
    def __init__(self,model_cls,model_dis,num_classes,num_disease):
        super(modelclassifierwithclip,self).__init__()
        self.model_cls = model_cls      
        self.model_dis = model_dis

        self.vision_width = 512
        self.layer_1 = 1
        self.heads_1 =  self.vision_width // 64

        # self.transformerc = Transformer(self.vision_width, self.layer_1, self.heads_1)
        self.transformerp = Transformer(self.vision_width, self.layer_1, self.heads_1)
        self.transformerd = Transformer(self.vision_width, self.layer_1, self.heads_1)

        self.layerNorm1 = LayerNorm(self.vision_width)
        self.layerNorm2 = LayerNorm(self.vision_width)
        
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        
        self.linear_c = nn.Linear((self.vision_width), num_classes)
        self.linear_d = nn.Linear((self.vision_width), num_disease)
        
        scale = self.vision_width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(self.vision_width*2, self.vision_width))
        
        self.transformerp.apply(self._init_weights)
        self.transformerd.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        
    def forward(self, PI, PT, DI, DT):

        self.p_image_f, self.d_image_f, self.c_image_c = self.image_features_base(PI , DI)
        self.p_text_f, self.d_text_f, self.c_text_c = self.text_features_base(PT, DT)
        self.p_image_f = self.p_image_f.permute(1, 0, 2)
        self.d_image_f = self.d_image_f.permute(1, 0, 2)

        # self.c_image_c = self.transformerc(self.c_image_c)

        self.p_image_f2 = self.transformerp(self.c_image_c)
        self.p_image_f2 = self.p_image_f2.permute(1, 0, 2)
        self.p_image_f2 = self.layerNorm1(self.p_image_f2[:, 0, :] + self.p_image_f[:, 0, :])
        self.p_output = self.linear_c(self.gelu1(self.p_image_f2))
        
        self.d_image_f2 = self.transformerd(self.c_image_c)
        self.d_image_f2 = self.d_image_f2.permute(1, 0, 2)
        self.d_image_f2 = self.layerNorm2(self.d_image_f2[:, 0, :] + self.d_image_f[:, 0, :])
        self.d_output = self.linear_d(self.gelu2(self.d_image_f2))        
    
        return (self.p_output, self.d_output, self.p_image_f2, self.d_image_f2, self.p_text_f, self.d_text_f)

    def image_features_base(self, PI , DI):
        self.p_image_f  = self.model_cls.forward_image_clip_features(PI)
        self.d_image_f  = self.model_dis.forward_image_clip_features(DI) 

        #Summation
        # self.c_image_c = self.p_image_f + self.d_image_f 
        
        #Concatenation
        self.c_image_c = torch.cat((self.p_image_f,self.d_image_f),-1)
        
        #Multiplication
        # self.c_image_c = torch.mul(self.p_image_f, self.d_image_f)
        
        self.c_image_c = self.c_image_c @ self.proj
        
        return (self.p_image_f, self.d_image_f, self.c_image_c)

    def text_features_base(self, PT, DT):
        self.p_text_f  = self.model_cls.encode_text_features(PT)
        self.d_text_f  = self.model_dis.encode_text_features(DT) 
        
        self.p_text_f = self.p_text_f.permute(1, 0, 2)
        self.d_text_f = self.d_text_f.permute(1, 0, 2)
        self.p_text_f = self.p_text_f[torch.arange(self.p_text_f.shape[0]), PT.argmax(dim=-1)]
        self.d_text_f = self.d_text_f[torch.arange(self.d_text_f.shape[0]), DT.argmax(dim=-1)]
        self.p_text_f = self.p_text_f / self.p_text_f.norm(dim=1, keepdim=True)
        self.d_text_f = self.d_text_f / self.d_text_f.norm(dim=1, keepdim=True)
        self.c_text_c = torch.cat((self.p_text_f,self.d_text_f),-1)

        return (self.p_text_f, self.d_text_f, self.c_text_c)

    def text_features_base_test(self, PT, DT):
        self.p_text_f  = self.model_cls.encode_text_abel(PT)
        self.d_text_f  = self.model_dis.encode_text_abel(DT) 
        
        self.p_text_f = self.p_text_f.permute(1, 0, 2)
        self.d_text_f = self.d_text_f.permute(1, 0, 2)
        self.p_text_f = self.p_text_f[:, 0, :]
        self.d_text_f = self.d_text_f[:, 0, :]
        self.p_text_f = self.p_text_f / self.p_text_f.norm(dim=1, keepdim=True)
        self.d_text_f = self.d_text_f / self.d_text_f.norm(dim=1, keepdim=True)
        
        return (self.p_text_f, self.d_text_f)

def main():

    # Data path for datasets and metadatas
    data_path = 'D:/DPD_dataset/PV/plantvillage_ori/plantvillage_(mix)'
    Save_model_path = "./save model"
    # MODEL_PATH = "path to your pretrained weight in pth"
    MODEL_PATH = "C:/Users/abel_/Desktop/Deep_learning/model_weight/pretrained/CLIP_ori_weight.pth"
    
    train_plant_list = './csv_ffclip/Train_plant'
    train_disease_list = './csv_ffclip/Train_disease'
    
    test_csv_path1 = "./csv_ffclip/PV_test.csv"
    test_csv_path2 = "./csv_ffclip/PV_unseen_test.csv"
    
    os.makedirs(Save_model_path, exist_ok=True)
    
    # Variables
    img_size = 224
    num_plant = 14
    num_disease = 21
    batch_size = 15
    num_epochs = 30
    learning_rate = 0.001
    weight_decay = 0.00001
    momentum = 0.9
    num_workers = 4
    pretrained = True
    grad_accumulation = True
    
    # Choose computation device
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print("device = ", device)
    model_name = "CLIP"
    
    print('Creating customised CLIP')
    model_cls = CLIP(
                embed_dim = 512,
                image_resolution = 224,
                vision_layers = 12,
                vision_width = 768,
                vision_patch_size = 32,
                context_length = 77,
                vocab_size = 49408,
                transformer_width = 512,
                transformer_heads = 8,
                transformer_layers = 12)
    model_dis = CLIP(
                embed_dim = 512,
                image_resolution = 224,
                vision_layers = 12,
                vision_width = 768,
                vision_patch_size = 32,
                context_length = 77,
                vocab_size = 49408,
                transformer_width = 512,
                transformer_heads = 8,
                transformer_layers = 12)
    
    
    if pretrained:
        print("Using Pre-Trained Plant Model")
        # model_cls = torch.jit.load(MODEL_PATH)
        model_cls.load_state_dict(torch.load(MODEL_PATH),strict=True)
    model_cls.to(device)
    
    if pretrained:
        print("Using Pre-Trained Disease Model")
        # model_dis = torch.jit.load(MODEL_PATH)
        model_dis.load_state_dict(torch.load(MODEL_PATH),strict=True)
    model_dis.to(device)
    
    test_transforms = Compose([
                Resize(img_size, img_size),
                CenterCrop(img_size, img_size, p=1.),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
    
    # train_dataset = CustomDatasetForTrainFFCLIP(train_plant_list, train_disease_list, transforms = test_transforms, data_path = data_path)
    train_dataset = CustomDatasetForTrainFFCLIP(train_plant_list, train_disease_list, data_path, data_path, transform = test_transforms)
    test_dataset1 = CustomDatasetForTestFFCLIP(test_csv_path1, transforms = test_transforms, data_path = data_path)
    test_dataset2 = CustomDatasetForTestFFCLIP(test_csv_path2, transforms = test_transforms, data_path = data_path)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last = False)
    test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last = False)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last = False)
    
    # Specify the loss function for images
    loss_img = nn.CrossEntropyLoss()
    
    loss_function = nn.CosineEmbeddingLoss()
    
    model_classifier = modelclassifierwithclip(model_cls, model_dis, num_plant, num_disease)
    convert_weights(model_classifier)
    model_classifier.to(device)
    
    # Prepare the optimizer
    optimizer_classifier = torch.optim.SGD(model_classifier.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum = momentum)
    
    # if pretrained:
    #     print("Using Pre-Trained Model for FF-CLIP")
    #     MODEL_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/CLIP/CLIP_16e_0.001L_WAug_100.0000_33.7963_FFCLIP_0C1PD_cat_1.0+0.5_120b_PV_37c_2Out-classifier_S0-1.pth"
    #     model_classifier.load_state_dict(torch.load(MODEL_PATH),strict=True)
    
    # if pretrained:
    #     print("Using Pre-Trained optimizer")
    #     OPTIMIZER_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/CLIP/CLIP_16e_0.001L_WAug_100.0000_33.7963_FFCLIP_0C1PD_cat_1.0+0.5_120b_PV_37c_2Out-opt_S0-1.pth"
    #     optimizer_classifier.load_state_dict(torch.load(OPTIMIZER_PATH))
    
    for g in optimizer_classifier.param_groups:
        g['lr'] = learning_rate
        
    tb = SummaryWriter()
    
    # For debug
    
    label_combine_train_list_full = []
    predictions_test_cls_list_full = []
    predictions_test_dis_list_full = []
    predictions_test_cls_dis_list_full = []
    predictions_test_combine_list_full = []
    
    for epoch in range(num_epochs):
        print(f"\nStart of Epoch {epoch+1} of {num_epochs}")
        print('Current Learning rate: {0}'.format(optimizer_classifier.param_groups[0]['lr']))
        total_train = 0
        correct_train = 0
        correct_train_cls = 0
        correct_train_dis = 0
        correct_train_cls_dis = 0
        
        total_loss_CE_p = 0
        total_loss_CE_d = 0 
        total_loss_CE_pd = 0
        total_loss_COS_p = 0
        total_loss_COS_d = 0
        total_loss_COS_pd = 0
        total_loss_combine = 0
        
        iter_batch = 8
        # iter_batch = 42
    
        model_cls.train()
        model_dis.train()
        model_classifier.train()
        train_dataset.set_epoch(epoch)
        if grad_accumulation:
            print(f"Gradient accumulation with total batch size: {batch_size*iter_batch}")
            
        for batch_idx, (images_p, labels_cls, text_p, images_d, labels_dis, text_d) in enumerate(tqdm(train_loader)):
            images_p, labels_cls, text_p = images_p.to(device), labels_cls.to(device), text_p.to(device)
            images_d, labels_dis, text_d = images_d.to(device), labels_dis.to(device), text_d.to(device)        
    
            output_p, output_d, output_pf, output_df, output_pt, output_dt = model_classifier(images_p, text_p, images_d, text_d)
            output_pf = output_pf / output_pf.norm(dim=1, keepdim=True)
            output_df = output_df / output_df.norm(dim=1, keepdim=True)
            
            lossCE_p = loss_img(output_p, labels_cls)
            total_loss_CE_p += lossCE_p.item()
            
            lossCE_d = loss_img(output_d, labels_dis)
            total_loss_CE_d += lossCE_d.item()        
    
            
            lossCOS_p = loss_function(output_pf, output_pt, Variable(torch.Tensor(output_p.size(0)).cuda().fill_(1.0)))
            total_loss_COS_p += lossCOS_p.item()
            lossCOS_d = loss_function(output_df, output_dt, Variable(torch.Tensor(output_d.size(0)).cuda().fill_(1.0)))
            total_loss_COS_d += lossCOS_d.item()
            
            loss_combine = 1.0*(lossCE_p + lossCE_d) + 1.0*(lossCOS_p + lossCOS_d)        
            total_loss_combine += loss_combine.item() 

            if grad_accumulation:
                loss_combine = loss_combine / iter_batch
                loss_combine.backward()
                if ((batch_idx + 1) % iter_batch == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer_classifier.step()
                    optimizer_classifier.zero_grad()
            else:
                loss_combine.backward()
                optimizer_classifier.step()
                optimizer_classifier.zero_grad()               
                
            predictions_cls = torch.max(output_p, 1)[1].to(device)
            correct_train_cls += (predictions_cls == labels_cls).sum()
            
            predictions_dis = torch.max(output_d, 1)[1].to(device)
            correct_train_dis += (predictions_dis == labels_dis).sum() 
    
            
            for x in range(len(labels_cls)):
                if (predictions_cls[x] == labels_cls[x]):
                    if (predictions_dis[x] == labels_dis[x]):
                        correct_train += 1
    
            total_train += len(labels_cls)
    
        l1 = total_loss_CE_p / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        l2 = total_loss_CE_d / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        l3 = total_loss_COS_p / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        l4 = total_loss_COS_d / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        l5 = total_loss_CE_pd / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        l6 = total_loss_COS_pd / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        l_total = total_loss_combine / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        tb.add_scalar("Training Plant CE Loss per epoch", l1 , epoch)
        tb.add_scalar("Training Disease CE Loss per epoch", l2 , epoch)
        tb.add_scalar("Training Plant Disease CE Loss per epoch", l5 , epoch)
        tb.add_scalar("Training Plant COS Loss per epoch", l3 , epoch)
        tb.add_scalar("Training Disease COS Loss per epoch", l4 , epoch)
        tb.add_scalar("Training Plant Disease COS Loss per epoch", l6 , epoch)
        tb.add_scalar("Total Training Loss per epoch", l_total , epoch)
    
        accuracy_train = correct_train * 100 / total_train
        accuracy_train_cls = correct_train_cls * 100 / total_train
        accuracy_train_dis = correct_train_dis * 100 / total_train
    
        print('Plant acc: {:.4f}'.format(accuracy_train_cls))
        print('Disease acc: {:.4f}'.format(accuracy_train_dis))
        print('Plant disease acc: {:.4f}'.format(accuracy_train))
        print('Plant CE loss: {0}'.format(l1))
        print('Disease CE loss: {0}'.format(l2))
        print('Plant Disease CE loss: {0}'.format(l5))
        print('Plant COS loss: {0}'.format(l3))
        print('Disease COS loss: {0}'.format(l4))
        print('Plant Disease COS loss: {0}'.format(l6))
        print('Total loss: {0}'.format(l_total))
    
        print(f"\nEpoch {epoch+1} of {num_epochs} Done!")
    
        print(f"\nSeen")
        total_test = 0
        correct_test_cls = 0
        correct_test_dis = 0
        correct_test_cls_dis = 0
        correct_test_combine = 0
        model_cls.eval()
        model_dis.eval()
        model_classifier.eval()
        
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_loss_combine = 0
        
        for batch_idx, (images_p, labels_cls, text_p, labels_dis, text_d) in enumerate(tqdm(test_loader1)):
            images_p, labels_cls, text_p = images_p.to(device), labels_cls.to(device), text_p.to(device)
            labels_dis, text_d = labels_dis.to(device), text_d.to(device)
            
            output_p, output_d, output_pf, output_df, output_pt, output_dt = model_classifier(images_p, text_p, images_p, text_d)

            loss1 = loss_img(output_p, labels_cls)
            total_loss1 += loss1.item()
            loss2 = loss_img(output_d, labels_dis)
            total_loss2 += loss2.item()
    
            loss_combine = loss1 + loss2        
            total_loss_combine += loss_combine.item()
    
            predictions_test_cls = torch.max(output_p, 1)[1].to(device)
            predictions_test_cls_list = predictions_test_cls.cpu().numpy()
            predictions_test_cls_list_full = np.append(predictions_test_cls_list_full,predictions_test_cls_list)
            correct_test_cls += (predictions_test_cls == labels_cls).sum()       
    
            predictions_test_dis = torch.max(output_d, 1)[1].to(device)
            predictions_test_dis_list = predictions_test_dis.cpu().numpy()
            predictions_test_dis_list_full = np.append(predictions_test_dis_list_full,predictions_test_dis_list)
            correct_test_dis += (predictions_test_dis == labels_dis).sum() 
            
            for x in range(len(labels_cls)):
                if (predictions_test_cls[x] == labels_cls[x]):
                    if (predictions_test_dis[x] == labels_dis[x]):
                        correct_test_combine += 1
    
            total_test += len(labels_cls)
    
        c = total_loss1 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        d = total_loss2 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        cd = total_loss3 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        tcd = total_loss_combine / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        tb.add_scalar("Testing Seen Plant CE Loss per epoch", c , epoch)
        tb.add_scalar("Testing Seen Disease CE Loss per epoch", d , epoch)
        tb.add_scalar("Testing Seen Plant Disease CE Loss per epoch", cd , epoch)
        tb.add_scalar("Total Testing Seen CE Loss per epoch", tcd , epoch)
        # tb.close()         
        accuracy_test_cls  = correct_test_cls * 100 / total_test
        accuracy_test_dis  = correct_test_dis * 100 / total_test
        accuracy_test_combine  = correct_test_combine * 100 / total_test
        print('Testing Seen Plant acc: {:.4f}'.format(accuracy_test_cls))
        print('Testing Seen Disease acc: {:.4f}'.format(accuracy_test_dis))
        print('Testing Seen Plant disease acc: {:.4f}'.format(accuracy_test_combine))
        print('Testing Seen loss: {0}'.format(tcd))
    
        print(f"\nUnseen")
        total_test = 0
        correct_test_cls = 0
        correct_test_dis = 0
        correct_test_cls_dis = 0    
        correct_test_combine = 0
        model_cls.eval()
        model_dis.eval()
        model_classifier.eval()
    
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_loss_combine = 0
    
        for batch_idx, (images_p, labels_cls, text_p, labels_dis, text_d) in enumerate(tqdm(test_loader2)):
            images_p, labels_cls, text_p = images_p.to(device), labels_cls.to(device), text_p.to(device)
            labels_dis, text_d = labels_dis.to(device), text_d.to(device)
    
            output_p, output_d, output_pf, output_df, output_pt, output_dt = model_classifier(images_p, text_p, images_p, text_d)

            loss1 = loss_img(output_p, labels_cls)
            total_loss1 += loss1.item()
            loss2 = loss_img(output_d, labels_dis)
            total_loss2 += loss2.item()
    
            loss_combine = loss1 + loss2        
            total_loss_combine += loss_combine.item()
    
            predictions_test_cls = torch.max(output_p, 1)[1].to(device)
            predictions_test_cls_list = predictions_test_cls.cpu().numpy()
            predictions_test_cls_list_full = np.append(predictions_test_cls_list_full,predictions_test_cls_list)
            correct_test_cls += (predictions_test_cls == labels_cls).sum()       
    
            predictions_test_dis = torch.max(output_d, 1)[1].to(device)
            predictions_test_dis_list = predictions_test_dis.cpu().numpy()
            predictions_test_dis_list_full = np.append(predictions_test_dis_list_full,predictions_test_dis_list)
            correct_test_dis += (predictions_test_dis == labels_dis).sum() 
            
            for x in range(len(labels_cls)):
                if (predictions_test_cls[x] == labels_cls[x]):
                    if (predictions_test_dis[x] == labels_dis[x]):
                        correct_test_combine += 1
    
            total_test += len(labels_cls)
    
        c = total_loss1 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        d = total_loss2 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        cd = total_loss3 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        tcd = total_loss_combine / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
        tb.add_scalar("Testing Unseen Plant Loss per epoch", c , epoch)
        tb.add_scalar("Testing Unseen Disease Loss per epoch", d , epoch)
        tb.add_scalar("Testing Unseen Plant Disease Loss per epoch", cd , epoch)    
        tb.add_scalar("Total Testing Unseen Loss per epoch", tcd , epoch)
        tb.close()         
        accuracy_test_cls  = correct_test_cls * 100 / total_test
        accuracy_test_dis  = correct_test_dis * 100 / total_test
        accuracy_test_combine  = correct_test_combine * 100 / total_test
        print('Testing Unseen Plant acc: {:.4f}'.format(accuracy_test_cls))
        print('Testing Unseen Disease acc: {:.4f}'.format(accuracy_test_dis))
        print('Testing Unseen Plant disease acc: {:.4f}'.format(accuracy_test_combine))
        print('Testing Unseen loss: {0}'.format(tcd))
    
    
        if ((epoch+1) % 1 == 0):
            print("Saving Model")
            torch.save(model_classifier.state_dict(), os.path.join(Save_model_path,'{}_{}e_{}L_WAug_{:.4f}_{:.4f}_FFCLIP_2Out-classifier.pth'
                                                                    .format(model_name,epoch+1,learning_rate,accuracy_train,accuracy_test_combine)))
            print("Saving optimizer")
            torch.save(optimizer_classifier.state_dict(), os.path.join(Save_model_path,'{}_{}e_{}L_WAug_{:.4f}_{:.4f}_FFCLIP_2Out-opt.pth'
                                                                        .format(model_name,epoch+1,learning_rate,accuracy_train,accuracy_test_combine)))
            print("Saving done")
    
    print("Training done")  

if __name__ == "__main__":
    main()






