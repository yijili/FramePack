import os
import sys
import time
import argparse
import logging
import uuid
from typing import Optional
from datetime import datetime
import asyncio
import threading
import signal
import atexit
import gc
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
import einops
import safetensors.torch as sf
import math

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置北京时间
import time
import logging.handlers

class BeijingTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = time.localtime(record.created)
        # 转换为北京时间 (UTC+8)
        beijing_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(record.created + 8 * 3600))
        return beijing_time

# 重新设置日志格式，使用北京时间
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# 创建新的处理器并设置北京时间格式
handler = logging.StreamHandler()
formatter = BeijingTimeFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 全局变量
device = None
weight_dtype = torch.float16

# 延迟加载模型，避免在模块导入时加载
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
image_encoder = None
feature_extractor = None
transformer = None

# 服务相关全局变量
server_thread = None
should_exit = False

# 任务管理相关
active_tasks = {}  # 存储正在进行的任务
task_lock = threading.Lock()  # 任务锁

class TaskStatus:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "running"  # running, completed, failed, cancelled
        self.progress = 0
        self.message = ""
        self.result = None
        self.thread = None  # 添加线程引用

def cleanup_models():
    """清理模型以释放内存"""
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, image_encoder, feature_extractor, transformer
    logger.info("清理模型内存...")
    
    # 删除模型引用
    if text_encoder:
        del text_encoder
    if text_encoder_2:
        del text_encoder_2
    if tokenizer:
        del tokenizer
    if tokenizer_2:
        del tokenizer_2
    if vae:
        del vae
    if image_encoder:
        del image_encoder
    if feature_extractor:
        del feature_extractor
    if transformer:
        del transformer
        
    text_encoder = None
    text_encoder_2 = None
    tokenizer = None
    tokenizer_2 = None
    vae = None
    image_encoder = None
    feature_extractor = None
    transformer = None
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("模型内存清理完成")

def load_models():
    """加载所有模型"""
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, image_encoder, feature_extractor, transformer, device
    
    # 使用 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"使用设备: {device}")
    logger.info("开始加载模型...")
    
    # 导入必要的模块
    from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import SiglipImageProcessor, SiglipVisionModel
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    
    # 加载模型
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    
    # 设置模型状态
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()
    
    # 启用VAE切片和瓦片处理
    vae.enable_slicing()
    vae.enable_tiling()
    
    # 设置模型精度
    transformer.high_quality_fp32_output_for_inference = True
    logger.info('transformer.high_quality_fp32_output_for_inference = True')
    
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)
    
    # 冻结模型参数
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # 将模型移动到设备
    text_encoder.to(device)
    text_encoder_2.to(device)
    image_encoder.to(device)
    vae.to(device)
    transformer.to(device)
    
    logger.info("模型加载完成")
    if torch.cuda.is_available():
        logger.info(f"GPU内存状态: 已分配 {torch.cuda.memory_allocated()/1024**2:.1f} MB, 已缓存 {torch.cuda.memory_reserved()/1024**2:.1f} MB")

# 注册退出处理函数
atexit.register(cleanup_models)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的初始化代码
    try:
        load_models()
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise
    yield
    # 关闭时的清理代码
    logger.info("正在关闭应用...")
    cleanup_models()
    logger.info("应用已关闭")

app = FastAPI(title="FramePack API", description="FramePack FastAPI 服务，提供图像到视频生成接口", lifespan=lifespan)

class InferenceRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 31337
    total_second_length: float = 5.0
    latent_window_size: int = 9
    steps: int = 25
    cfg: float = 1.0
    gs: float = 10.0
    rs: float = 0.0
    gpu_memory_preservation: float = 6.0
    use_teacache: bool = True
    mp4_crf: int = 16
    output_path: Optional[str] = None
    task_id: Optional[str] = None

class InferenceResponse(BaseModel):
    task_id: str
    message: str
    video_path: Optional[str] = None

class TaskStatusRequest(BaseModel):
    task_id: str

class TaskResultRequest(BaseModel):
    task_id: str

class CancelTaskRequest(BaseModel):
    task_id: str

def generate_unique_filename(extension: str) -> str:
    """生成带时间戳的唯一文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.{extension}"

def check_task_cancelled(task_id: str) -> bool:
    """检查任务是否被取消"""
    with task_lock:
        if task_id in active_tasks:
            return active_tasks[task_id].status == "cancelled"
    return False

def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2):
    """编码提示词条件"""
    from diffusers_helper.hunyuan import encode_prompt_conds as hunyuan_encode_prompt_conds
    return hunyuan_encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

def vae_encode(image_tensor, vae):
    """VAE编码"""
    from diffusers_helper.hunyuan import vae_encode as hunyuan_vae_encode
    return hunyuan_vae_encode(image_tensor, vae)

def hf_clip_vision_encode(image_np, feature_extractor, image_encoder):
    """CLIP视觉编码"""
    from diffusers_helper.clip_vision import hf_clip_vision_encode as clip_vision_encode
    return clip_vision_encode(image_np, feature_extractor, image_encoder)

def crop_or_pad_yield_mask(tensor, length=512):
    """裁剪或填充张量"""
    from diffusers_helper.utils import crop_or_pad_yield_mask as utils_crop_or_pad_yield_mask
    return utils_crop_or_pad_yield_mask(tensor, length=length)

def resize_and_center_crop(image_np, target_width, target_height):
    """调整图像大小并居中裁剪"""
    from diffusers_helper.utils import resize_and_center_crop as utils_resize_and_center_crop
    return utils_resize_and_center_crop(image_np, target_width=target_width, target_height=target_height)

def find_nearest_bucket(height, width, resolution=640):
    """查找最近的桶尺寸"""
    from diffusers_helper.bucket_tools import find_nearest_bucket as bucket_find_nearest_bucket
    return bucket_find_nearest_bucket(height, width, resolution=resolution)

def sample_hunyuan(transformer, sampler, width, height, frames, real_guidance_scale, distilled_guidance_scale, 
                   guidance_rescale, num_inference_steps, generator, prompt_embeds, prompt_embeds_mask, 
                   prompt_poolers, negative_prompt_embeds, negative_prompt_embeds_mask, negative_prompt_poolers, 
                   device, dtype, image_embeddings, latent_indices, clean_latents, clean_latent_indices, 
                   clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices, callback):
    """采样Hunyuan模型"""
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan as k_diffusion_sample_hunyuan
    return k_diffusion_sample_hunyuan(
        transformer=transformer, sampler=sampler, width=width, height=height, frames=frames,
        real_guidance_scale=real_guidance_scale, distilled_guidance_scale=distilled_guidance_scale,
        guidance_rescale=guidance_rescale, num_inference_steps=num_inference_steps, generator=generator,
        prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask, prompt_poolers=prompt_poolers,
        negative_prompt_embeds=negative_prompt_embeds, negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        negative_prompt_poolers=negative_prompt_poolers, device=device, dtype=dtype,
        image_embeddings=image_embeddings, latent_indices=latent_indices, clean_latents=clean_latents,
        clean_latent_indices=clean_latent_indices, clean_latents_2x=clean_latents_2x,
        clean_latent_2x_indices=clean_latent_2x_indices, clean_latents_4x=clean_latents_4x,
        clean_latent_4x_indices=clean_latent_4x_indices, callback=callback
    )

def vae_decode_fake(latents):
    """VAE假解码"""
    from diffusers_helper.hunyuan import vae_decode_fake as hunyuan_vae_decode_fake
    return hunyuan_vae_decode_fake(latents)

def vae_decode(latents, vae):
    """VAE解码"""
    from diffusers_helper.hunyuan import vae_decode as hunyuan_vae_decode
    return hunyuan_vae_decode(latents, vae)

def save_bcthw_as_mp4(tensor, filename, fps=30, crf=16):
    """保存张量为MP4视频"""
    from diffusers_helper.utils import save_bcthw_as_mp4 as utils_save_bcthw_as_mp4
    return utils_save_bcthw_as_mp4(tensor, filename, fps=fps, crf=crf)

@torch.no_grad()
def inference_api(
    task_id: str,
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 31337,
    total_second_length: float = 5.0,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 10.0,
    rs: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = True,
    mp4_crf: int = 16,
    output_path: Optional[str] = None
) -> dict:
    """API 版本的 inference 函数，支持任务中断"""
    try:
        logger.info(f"开始执行 inference，任务ID: {task_id}，图像路径: {image_path}")
        
        # 确保模型已加载
        if not all([text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, image_encoder, feature_extractor, transformer]):
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
        
        # 设置参数
        result_dir = output_path if output_path else './results/output'
        os.makedirs(result_dir, exist_ok=True)
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 5
                active_tasks[task_id].message = "读取图像..."
        
        ############################################## 读取输入图像 ##############################################
        logger.info("读取输入图像...")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        input_image = Image.open(image_path).convert("RGB")
        input_image_np = np.array(input_image)
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 10
                active_tasks[task_id].message = "文本编码..."
        
        ############################################## 文本编码 ##############################################
        logger.info("文本编码...")
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 20
                active_tasks[task_id].message = "图像处理..."
        
        ############################################## 图像处理 ##############################################
        logger.info("图像处理...")
        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 30
                active_tasks[task_id].message = "VAE编码..."
        
        ############################################## VAE编码 ##############################################
        logger.info("VAE编码...")
        start_latent = vae_encode(input_image_pt, vae)
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 40
                active_tasks[task_id].message = "CLIP视觉编码..."
        
        ############################################## CLIP视觉编码 ##############################################
        logger.info("CLIP视觉编码...")
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 50
                active_tasks[task_id].message = "设置数据类型..."
        
        ############################################## 数据类型设置 ##############################################
        logger.info("设置数据类型...")
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 60
                active_tasks[task_id].message = "开始采样..."
        
        ############################################## 采样 ##############################################
        logger.info("开始采样...")
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # 在理论中，latent_paddings应该遵循上述序列，但似乎当total_latent_sections > 4时，
            # 重复某些项目比扩展它看起来更好
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            # 检查任务是否已取消
            if check_task_cancelled(task_id):
                logger.info(f"任务 {task_id} 已被取消")
                return {"message": "任务已被取消"}
                
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                # 检查任务是否已取消
                if check_task_cancelled(task_id):
                    raise KeyboardInterrupt('任务已被取消')
                    
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                current_step = d['i'] + 1
                percentage = int(60 + 20.0 * current_step / steps)  # 进度从60%到80%
                hint = f'采样中 {current_step}/{steps}'
                with task_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id].progress = percentage
                        active_tasks[task_id].message = hint

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=device,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                # 使用简单拼接而不是soft_append_bcthw以简化实现
                history_pixels = torch.cat([current_pixels, history_pixels], dim=2)

            # 检查任务是否已取消
            if check_task_cancelled(task_id):
                logger.info(f"任务 {task_id} 已被取消")
                return {"message": "任务已被取消"}
                
            # 更新任务状态
            progress_percent = int(80 + 15.0 * (total_latent_sections - latent_padding) / total_latent_sections)
            with task_lock:
                if task_id in active_tasks:
                    active_tasks[task_id].progress = progress_percent
                    active_tasks[task_id].message = f"生成视频帧... ({total_latent_sections - latent_padding}/{total_latent_sections})"

            output_filename = os.path.join(result_dir, generate_unique_filename("mp4"))
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            logger.info(f'解码完成。当前潜在空间形状 {real_history_latents.shape}; 像素形状 {history_pixels.shape}')

            if is_last_section:
                break
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 100
                active_tasks[task_id].status = "completed"
                active_tasks[task_id].message = "任务完成"
                active_tasks[task_id].result = {
                    "video_path": os.path.abspath(output_filename)
                }
        
        logger.info(f"结果已保存到: {output_filename}")
        return {
            "video_path": os.path.abspath(output_filename)
        }
        
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        logger.error(f"inference 执行出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时变量和强制垃圾回收
        try:
            if 'gen' in locals():
                del gen
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_inference_background(task_id: str, request: InferenceRequest):
    """在后台线程中运行推理任务"""
    try:
        result = inference_api(
            task_id=task_id,
            image_path=request.image_path,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            total_second_length=request.total_second_length,
            latent_window_size=request.latent_window_size,
            steps=request.steps,
            cfg=request.cfg,
            gs=request.gs,
            rs=request.rs,
            gpu_memory_preservation=request.gpu_memory_preservation,
            use_teacache=request.use_teacache,
            mp4_crf=request.mp4_crf,
            output_path=request.output_path
        )
        logger.info(f"任务 {task_id} 推理完成")
        
        # 更新任务状态为完成
        with task_lock:
            if task_id in active_tasks:
                if "message" in result and result["message"] == "任务已被取消":
                    active_tasks[task_id].status = "cancelled"
                    active_tasks[task_id].message = "任务已完成取消"
                else:
                    active_tasks[task_id].status = "completed"
                    active_tasks[task_id].result = result
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        
        logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
    finally:
        # 任务完成后强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/inference", response_model=InferenceResponse, summary="执行推理生成视频")
async def inference_endpoint(request: InferenceRequest):
    """
    执行完整的推理过程，从图像生成视频
    """
    logger.info("收到 inference 请求")
    
    # 检查图像文件是否存在
    if not os.path.exists(request.image_path):
        logger.error(f"图像文件不存在: {request.image_path}")
        raise HTTPException(status_code=404, detail=f"图像文件不存在: {request.image_path}")
    
    # 如果请求中包含task_id，则将其添加到active_tasks中
    task_id = request.task_id
    if not task_id:
        # 如果没有提供task_id，则生成一个新的
        task_id = str(uuid.uuid4())
    
    # 创建任务状态对象（如果需要跟踪进度）
    task_status = TaskStatus(task_id)
    
    # 将任务添加到活动任务列表
    with task_lock:
        active_tasks[task_id] = task_status
    
    # 在后台线程中运行推理任务
    thread = threading.Thread(target=run_inference_background, args=(task_id, request))
    thread.start()
    
    # 保存线程引用以便可能的管理
    with task_lock:
        active_tasks[task_id].thread = thread
    
    logger.info(f"任务 {task_id} 已启动后台推理线程")
    
    # 立即返回响应
    return InferenceResponse(
        task_id=task_id,
        message="任务已启动"
    )

@app.get("/task_status/{task_id}", summary="获取任务状态")
async def get_task_status(task_id: str):
    """
    获取指定任务的当前状态和进度
    """
    with task_lock:
        if task_id in active_tasks:
            task = active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "result": task.result
            }
        else:
            raise HTTPException(status_code=404, detail="任务不存在")

@app.post("/task_result", summary="等待并获取任务结果")
async def get_task_result(request: TaskResultRequest):
    """
    等待任务完成并返回最终结果
    """
    task_id = request.task_id
    logger.info(f"收到获取任务 {task_id} 结果的请求")
    
    # 首先检查任务是否存在
    with task_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # 等待任务完成
    while True:
        with task_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                # 如果任务已完成、失败或被取消，则退出循环
                if task.status in ["completed", "failed", "cancelled"]:
                    break
            else:
                raise HTTPException(status_code=404, detail="任务不存在")
        
        # 等待一段时间再检查
        await asyncio.sleep(1)
    
    # 返回任务结果
    with task_lock:
        task = active_tasks[task_id]
        if task.status == "completed":
            logger.info(f"任务 {task_id} 已完成，返回结果")
            return {
                "task_id": task_id,
                "status": task.status,
                "message": task.message,
                "result": task.result
            }
        elif task.status == "failed":
            logger.info(f"任务 {task_id} 执行失败")
            raise HTTPException(status_code=500, detail=task.message)
        elif task.status == "cancelled":
            logger.info(f"任务 {task_id} 已被取消")
            raise HTTPException(status_code=499, detail="任务已被取消")  # 499表示客户端关闭请求
        else:
            raise HTTPException(status_code=500, detail="未知任务状态")

@app.post("/cancel_task", summary="取消任务")
async def cancel_task(request: CancelTaskRequest):
    """
    取消指定的任务，并等待取消完成
    """
    task_id = request.task_id
    logger.info(f"收到取消任务 {task_id} 的请求")
    
    # 首先检查任务是否存在
    with task_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # 标记任务为取消状态
    with task_lock:
        active_tasks[task_id].status = "cancelled"
        active_tasks[task_id].message = "任务已被取消"
    
    # 等待任务真正完成取消
    while True:
        with task_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                # 如果任务已取消完成，则退出循环
                if task.status == "cancelled" and task.message == "任务已完成取消":
                    break
            else:
                # 任务已被完全清理
                break
        
        # 等待一段时间再检查
        await asyncio.sleep(0.5)
    
    logger.info(f"任务 {task_id} 取消完成")
    return {"message": "任务取消完成"}

@app.get("/", summary="API 根路径")
async def root():
    return {"message": "FramePack API 服务正在运行", 
            "endpoints": [
                "/inference",
                "/task_status/{task_id}", 
                "/task_result",
                "/cancel_task"
            ]}

@app.get("/results/{file_path:path}", summary="获取结果文件")
async def get_result_file(file_path: str):
    """
    提供对结果文件的访问
    """
    file_full_path = os.path.join("./results", file_path)
    if os.path.exists(file_full_path):
        return FileResponse(file_full_path)
    else:
        raise HTTPException(status_code=404, detail="文件未找到")

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在关闭服务...")
    global should_exit
    should_exit = True
    cleanup_models()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=6870, help="端口号")
    parser.add_argument("--isdebug", action="store_true", help="是否输出调试日志")
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 根据 isdebug 参数设置日志级别
    if args.isdebug:
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    logger.info(f"启动 FramePack API 服务，当前版本1.0.0，主机: {args.host}，端口: {args.port}")
    
    # 不使用 reload 参数，避免重复导入
    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)