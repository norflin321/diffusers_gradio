import torch
import diffusers as df
import const as const
import gc
import utils as utils
import controlnet_aux as cn
import huggingface_hub as hf
import safetensors as sf

class General:
  pipeline = None
  pipeline_id = None
  loaders = {}
  loaders_img = {}
  
  def __init__(self):
    self.loaders[const.SDXL] = self.load_sdxl_pipeline
    self.loaders_img[const.SDXL] = self.load_sdxl_sketch_pipeline
    self.loaders[const.SDXLL] = self.load_sdxll_pipeline
    self.loaders_img[const.SDXLL] = self.load_sdxll_sketch_pipeline
  
  def clear_pipeline(self):
    if self.pipeline is None: return
    utils.log("clear_pipeline", self.pipeline_id)
    self.pipeline = None
    self.pipeline_id = None
    torch.cuda.empty_cache()
    gc.collect()

  def handle_lora_finetune(self, has_lora, lora):
    utils.log("handle_lora_finetune", has_lora, lora)
    if has_lora: self.pipeline.load_lora_weights(const.LORA_DIR, weight_name=lora)
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
  
  def pidinet_sketch(self, img, res):
    pidinet = cn.pidi.PidiNetDetector.from_pretrained("lllyasviel/Annotators", cache_dir=const.CACHE_DIR).to("cuda")
    return pidinet(img, detect_resolution=res, image_resolution=res, apply_filter=True)

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

  def load_sdxll_pipeline(self):
    utils.log("load_sdxll_pipeline")
    unet = df.UNet2DConditionModel.from_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(sf.torch.load_file(hf.hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors", cache_dir=const.CACHE_DIR), device="cuda"))
    pipeline = df.StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipeline.scheduler = df.EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    return pipeline
  
  def load_sdxll_sketch_pipeline(self):
    utils.log("load_sdxll_sketch_pipeline")
    adapter = df.T2IAdapter.from_pretrained("TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")
    unet = df.UNet2DConditionModel.from_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(sf.torch.load_file(hf.hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors", cache_dir=const.CACHE_DIR), device="cuda"))
    pipeline = df.StableDiffusionXLAdapterPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipeline.scheduler = df.EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    return pipeline

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
    if is_pipeline_is_changed or self.has_lora != has_lora or self.low_vram != low_vram:
      self.clear_pipeline()
      self.pipeline_id = pipeline_id
      self.has_lora = has_lora
      self.low_vram = low_vram
      
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
    if model == const.SDXL and has_img: # SDXL with Sketch Adapter
      res = self.pipeline(prompt=txt, negative_prompt=n_txt, num_inference_steps=steps, guidance_scale=txt_guid, generator=gn, width=w, height=h, num_images_per_prompt=batch, image=self.pidinet_sketch(img, w), adapter_conditioning_scale=img_guid).images
    elif model == const.SDXLL and has_img: # SDXL Lightning with Sketch Adapter
      res = self.pipeline(prompt=txt, negative_prompt=n_txt, num_inference_steps=4, guidance_scale=0, generator=gn, width=w, height=h, num_images_per_prompt=batch, image=self.pidinet_sketch(img, w), adapter_conditioning_scale=img_guid).images
    elif model == const.SDXLL and not has_img: # SDXL Lightning
      res = self.pipeline(prompt=txt, negative_prompt=n_txt, num_inference_steps=4, guidance_scale=0, generator=gn, width=w, height=h, num_images_per_prompt=batch).images
    else:
      res = self.pipeline(prompt=txt, negative_prompt=n_txt, num_inference_steps=steps, guidance_scale=txt_guid, generator=gn, width=w, height=h, num_images_per_prompt=batch).images
    
    # handle result
    utils.log("done")
    utils.save_imgs(res, seed)
    return [res, seed]