import torch
import diffusers as df
import const as const
import gc
import utils as utils
import controlnet_aux as cn

class General:
  def clear_pipeline(self):
    print("clear_pipeline", self.pipeline_id)
    self.pipeline = None
    self.pipeline_id = None
    torch.cuda.empty_cache()
    gc.collect()
  
  def handle_lora_finetune(self, is_use_lora, lora_finetune):
    print("handle_lora_finetune", is_use_lora, lora_finetune)
    if is_use_lora:
      self.pipeline.load_lora_weights(const.LORA_FINETUNES_PATH, weight_name=lora_finetune)
    else:
      self.pipeline.unload_lora_weights()
  
  def optimize_vram(self):
    print("optimize_vram")
    self.pipeline.enable_model_cpu_offload()
    # self.pipeline.enable_xformers_memory_efficient_attention()
    
  def load_generic_pipeline(self, model_name):
    print("load_generic_pipeline", model_name)
    self.pipeline = df.DiffusionPipeline.from_pretrained(model_name, use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
    
  def load_sdxl_pipeline(self):
    print("load_sdxl_pipeline", const.SDXL_MODEL_NAME)
    euler_a = df.EulerAncestralDiscreteScheduler.from_pretrained(const.SDXL_MODEL_NAME, subfolder="scheduler", use_safetensors=True)
    vae = df.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    self.pipeline = df.StableDiffusionXLPipeline.from_pretrained(const.SDXL_MODEL_NAME, vae=vae, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16")
    
  def load_sdxl_sketch_pipeline(self, model_name):
    print("load_sdxl_sketch_pipeline")
    adapter = df.T2IAdapter.from_pretrained("TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")
    euler_a = df.EulerAncestralDiscreteScheduler.from_pretrained(const.SDXL_MODEL_NAME, subfolder="scheduler", use_safetensors=True)
    vae = df.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    self.pipeline = df.StableDiffusionXLAdapterPipeline.from_pretrained(model_name, adapter=adapter, scheduler=euler_a, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
  
  def generate(self, img, img_guid, model, prompt, n_prompt, width, height, steps, prompt_guid, batch, seed, lora, is_low_vram):
    print("generate")
    pipeline_id = model
    if img is not None: pipeline_id += "_prompt_image"
      
    is_use_lora = lora and lora != ""
    
    # do we need to load new pipeline?
    if self.pipeline_id != pipeline_id or self.is_use_lora != is_use_lora or self.is_low_vram != is_low_vram:
      self.clear_pipeline()
      self.pipeline_id = pipeline_id
      self.is_use_lora = is_use_lora
      self.is_low_vram = is_low_vram
      
      # load pipeline
      if model == const.SDXL_MODEL_NAME:
        if img is None:
          self.load_sdxl_pipeline()
        else:
          self.load_sdxl_sketch_pipeline()
      else:
        self.load_generic_pipeline(model)

      # pipeline optimizations
      self.pipeline.to("cuda")
      if is_low_vram: self.optimize_vram()
    
    self.handle_lora_finetune(is_use_lora, lora)
    
    # handle seed
    if seed is None or seed <= 0: seed = None
    current_seed = seed or torch.randint(0, int(1e5), size=(1, 1))[0].item()
    gn = torch.Generator().manual_seed(int(current_seed))
    
    # run pipeline
    if img is not None:
      # SDXL with Sketch adapter
      pidinet = cn.pidi.PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
      image = pidinet(img, detect_resolution=width, image_resolution=width, apply_filter=True)
      result_images = self.pipeline(prompt=prompt, image=image, negative_prompt=n_prompt, num_inference_steps=steps, guidance_scale=prompt_guid, generator=gn, width=int(width), height=int(height), adapter_conditioning_scale=img_guid, num_images_per_prompt=batch).images
    else:
      result_images = self.pipeline(prompt=prompt, negative_prompt=n_prompt, num_inference_steps=steps, guidance_scale=prompt_guid, generator=gn, width=int(width), height=int(height), num_images_per_prompt=batch).images
    
    # handle generation results
    print("done")
    for image in result_images: utils.save_result_img(image, current_seed)
    return [result_images, current_seed]