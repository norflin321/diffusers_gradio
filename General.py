import torch
import diffusers as df
import const as const
import gc
import utils as utils
import controlnet_aux as cn

class General:
  pipeline_id = None
  loaders = {}
  loaders_img = {}
  
  def __init__(self):
    self.loaders[const.SDXL] = self.load_sdxl_pipeline
    self.loaders_img[const.SDXL] = self.load_sdxl_sketch_pipeline
  
  def clear_pipeline(self):
    utils.log("clear_pipeline", self.pipeline_id)
    self.pipeline = None
    self.pipeline_id = None
    torch.cuda.empty_cache()
    gc.collect()

  def handle_lora_finetune(self, has_lora, lora):
    utils.log("handle_lora_finetune", has_lora, lora)
    if has_lora: self.pipeline.load_lora_weights(const.LORA_FINETUNES_PATH, weight_name=lora)
    else: self.pipeline.unload_lora_weights()

  def optimize_vram(self):
    utils.log("optimize_vram")
    self.pipeline.enable_model_cpu_offload()
    # self.pipeline.enable_xformers_memory_efficient_attention()
    
  def get_seed_gn(self, seed):
    utils.log("get_seed_gn", seed)
    if seed is None or seed <= 0:
      new_seed = torch.randint(0, int(1e5), size=(1, 1))[0].item()
      gn = torch.Generator().manual_seed(int(new_seed))
      return (new_seed, gn)
    gn = torch.Generator().manual_seed(int(seed))
    return (seed, gn)
  
  def get_pipeline_loader(self, model, has_img):
    utils.log("get_pipeline_loader")
    ld = self.loaders_img if has_img else self.loaders
    if model not in ld: return None
    return ld[model]
  
  def load_generic_pipeline(self, model):
    utils.log("load_generic_pipeline", model)
    return df.DiffusionPipeline.from_pretrained(model, use_safetensors=True, torch_dtype=torch.float16, variant="fp16")

  def load_sdxl_pipeline(self):
    utils.log("load_sdxl_pipeline")
    euler_a = df.EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", use_safetensors=True)
    vae = df.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    return df.StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16")

  def load_sdxl_sketch_pipeline(self):
    utils.log("load_sdxl_sketch_pipeline")
    adapter = df.T2IAdapter.from_pretrained("TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")
    euler_a = df.EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", use_safetensors=True)
    vae = df.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    return df.StableDiffusionXLAdapterPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, scheduler=euler_a, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

  def generate(self, img, model, lora, txt, n_txt, steps, txt_guid, img_guid, w, h, batch, seed, low_vram):
    utils.log("generate")
    has_img = bool(img is not None)
    has_lora = bool(lora and lora != "")
    w = int(w)
    h = int(h)
    
    # define pipeline id
    pipeline_id = model
    if has_img: pipeline_id += "_img"
    
    is_pipeline_is_changed = self.pipeline_id != pipeline_id
    
    # do we need to load new pipeline?
    if is_pipeline_is_changed or self.is_use_lora != has_lora or self.is_low_vram != low_vram:
      self.clear_pipeline()
      self.pipeline_id = pipeline_id
      self.is_use_lora = has_lora
      self.is_low_vram = low_vram
      
      # choose pipeline loader and run it
      loader = self.get_pipeline_loader(model, has_img)
      if loader is not None: self.pipeline = loader()
      else: self.pipeline = self.load_generic_pipeline(model)

      # pipeline optimizations
      self.pipeline.to("cuda")
      if low_vram: self.optimize_vram()
    
    self.handle_lora_finetune(has_lora, lora)
    (seed, gn) = self.get_seed_gn(seed)
    
    # run pipeline
    if has_img:
      # SDXL with Sketch adapter
      pidinet = cn.pidi.PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
      img = pidinet(img, detect_resolution=w, image_resolution=w, apply_filter=True)
      res = self.pipeline(prompt=txt, negative_prompt=n_txt, num_inference_steps=steps, guidance_scale=txt_guid, generator=gn, width=w, height=h, num_images_per_prompt=batch, image=img, adapter_conditioning_scale=img_guid).images
    else:
      res = self.pipeline(prompt=txt, negative_prompt=n_txt, num_inference_steps=steps, guidance_scale=txt_guid, generator=gn, width=w, height=h, num_images_per_prompt=batch).images
    
    # handle result
    utils.log("done")
    utils.save_imgs(res, seed)
    return [res, seed]