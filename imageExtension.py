from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from tqdm import tqdm
import cv2 as cv
import numpy as np
import time
from safetensors.torch import load_file
from typing import List



class ImageExtension():
    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-inpainting"
    cache_dir = "F:/huggingface_model/"    # diffusers的本地缓存路径
    def __init__(self, 
                 sampleTimeStep:int = 30, 
                 only_local_files:bool = False):
        self.base_model_name = self.DEFAULT_MODEL
        self.only_local_files = only_local_files
        self.sampleTimeStep = sampleTimeStep
        self.load_model()
        self.getLatent_model()
        self.load_text_encoder()
        self.load_scheduler()
        
        self.allTimestep = self.ddim.timesteps

        self.image_processor = VaeImageProcessor()


    def addNoise(self, latent:torch.Tensor, noise: torch.Tensor, timestep:torch.Tensor):
        latent = self.ddim.add_noise(latent, noise, timestep)
        latent = latent * self.ddim.init_noise_sigma
    
        return latent
    
    def set_step(self, sampleTimeStep:int):
        self.sampleTimeStep = sampleTimeStep
        self.ddim.set_timesteps(self.sampleTimeStep, device="cuda:0")
        self.allTimestep = self.ddim.timesteps
    
        
    def sample_step(self, latent: torch.Tensor, niose: torch.Tensor, timestep: torch.Tensor):
        return self.ddim.step(niose, timestep, latent)['prev_sample']


    def sample_block(self, latent:torch.Tensor, masked_latent:torch.Tensor, mask: torch.Tensor, prompt_embeds:torch.Tensor, timestep: torch.Tensor, guidance_scale:int=7):
        latent_model_input = torch.cat([latent] * 2)
        mask_input = torch.cat([mask]*2)
        masked_latent_input = torch.cat([masked_latent]*2)

        latent_model_input = self.ddim.scale_model_input(latent_model_input, timestep)
        latent_model_input = torch.cat([latent_model_input, mask_input, masked_latent_input], dim=1)     # inpaint模型拥有额外的输入信息，通道数为9
        
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states = prompt_embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latent = self.sample_step(latent, noise_pred, timestep)

        return latent
    
    def sample(self, latent:torch.Tensor, masked_latent:torch.Tensor, mask: torch.Tensor, chunk_list: List[int], prompt_embeds:torch.Tensor, guidance_scale:int=7):
        # print(prompt_embeds.shape)
        # print(latent.shape)
        count = torch.zeros_like(latent)
        full_latent = torch.zeros_like(latent)
        for Tin in tqdm(range(0, len(self.allTimestep))):
            Ti = self.allTimestep[Tin]
            count.zero_()
            full_latent.zero_()
            for chunk_block in chunk_list:

                sample_latent = latent[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]]
                sample_mask = mask[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]]
                sample_masked_latent = masked_latent[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]]

                pred_noise = self.sample_block(sample_latent, sample_masked_latent, sample_mask, prompt_embeds, Ti, guidance_scale)   # 每一个时间步的采样过程
                
                full_latent[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]] += pred_noise
                count[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]] += 1


            latent = torch.where(count > 0, full_latent / count, full_latent)

        return latent
    
    def load_scheduler(self):
        MN = self.base_model_name
        self.ddim = DDIMScheduler.from_pretrained(MN, 
                                                  subfolder="scheduler", 
                                                  local_files_only=self.only_local_files, 
                                                #   torch_dtype=torch.float16, 
                                                  use_safetensors=True, 
                                                  cache_dir = self.cache_dir)
        

        self.ddim.set_timesteps(self.sampleTimeStep, device="cuda:0")

    def load_model(self):
        self.unet = UNet2DConditionModel.from_pretrained(self.base_model_name, 
                                                            local_files_only = self.only_local_files, 
                                                            torch_dtype=torch.float16, 
                                                            # use_safetensors=True, 
                                                            subfolder = "unet",
                                                            cache_dir = self.cache_dir).cuda()
        
     
        self.unet.enable_xformers_memory_efficient_attention()
        
    def getLatent_model(self):
        MN = self.base_model_name
        self.vae = AutoencoderKL.from_pretrained(MN, 
                                                 local_files_only = self.only_local_files,
                                                 torch_dtype=torch.float16,
                                                #  use_safetensors=True,
                                                 subfolder = "vae",
                                                 cache_dir = self.cache_dir).cuda()
        

    def load_text_encoder(self):
        MN = self.base_model_name
        self.text_encoder = CLIPTextModel.from_pretrained(MN, 
                                                          local_files_only = self.only_local_files,
                                                          torch_dtype=torch.float16,
                                                        #   use_safetensors=True,
                                                          subfolder = "text_encoder",
                                                          cache_dir = self.cache_dir).cuda()
    
        self.tokenizer = CLIPTokenizer.from_pretrained(MN,
                                                         local_files_only = self.only_local_files,
                                                         subfolder = "tokenizer",
                                                         cache_dir = self.cache_dir)
        
        
    @staticmethod
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids 

    
    def encode_prompt(self, prompt:str, neg_prompt:str = None):
        text_input_ids = self.tokenize_prompt(self.tokenizer, prompt)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        encoder_hidden_states = prompt_embeds.hidden_states[-2]
        prompt_embeds = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
        # prompt_embeds = prompt_embeds[0]

        if neg_prompt is None:
            neg_prompt = ""
        negative_text_input_ids = self.tokenize_prompt(self.tokenizer, neg_prompt)
        negative_prompt_embeds = self.text_encoder(
            negative_text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
    
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def get_text_embedding(self, prompt:str, neg_prompt:str = None):
        return self.encode_prompt(prompt, neg_prompt)

        
    def getImgLatent(self, img:torch.Tensor):
        # img = self.image_processor.preprocess(img)
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
    
    def getImg(self, latent:torch.Tensor):
        image = self.vae.decode(latent / self.vae.config.scaling_factor)[0]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type="np", do_denormalize=[True])
        return image

    
def main():
    pass

if __name__ == "__main__":
    main()

