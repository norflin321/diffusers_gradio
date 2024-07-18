import diffusers as df
import gradio as gr
import torch
import controlnet_aux as cn
import os
import gc
import datetime

PWD = os.getcwd()

SDXL_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SUGGESTED_MODELS = [SDXL_MODEL_NAME, "your ass"]
WIDTH_CHOICES = [1024, 768, 512]
HEIGHT_CHOICES = [1024, 768, 512]
default_prompt = "a magic creature in style of sks, 3D, blender, perfect, 4k graphics, highly detailed, cute, pretty"
default_negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

# define some required folders and ensure they exist
LORA_FINETUNES_PATH = f"{PWD}/lora_finetunes"
if not os.path.exists(LORA_FINETUNES_PATH): os.makedirs(LORA_FINETUNES_PATH)
RESULTS_PATH = f"{PWD}/results"
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
MODELS_PATH = f"{PWD}/models"
if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)

def save_result_img(img, seed):
  time_now = datetime.datetime.now().strftime("%y.%m.%d_%H-%M-%S")
  img.save(f"{RESULTS_PATH}/{time_now}_{seed}.png")
  
def allow_only_sdxl_model():
  return gr.update(choices=[SDXL_MODEL_NAME], value=SDXL_MODEL_NAME, interactive=True)

def allow_all_models():
  return gr.update(choices=SUGGESTED_MODELS, interactive=True)

def update_lora_finetune_choices():
  choices = [f for f in os.listdir(LORA_FINETUNES_PATH) if os.path.isfile(os.path.join(LORA_FINETUNES_PATH, f))]
  choices.insert(0, "")
  return gr.update(choices=choices, interactive=True, value=None)

class Main:
  pipeline: df.DiffusionPipeline = None
  pipeline_id: str = None

  def generate(self, prompt_image, prompt_image_guidance, model_name, prompt, negative_prompt, width, height, steps, prompt_guidance, batch, seed, lora_finetune, is_low_vram):
    pipeline_id = model_name
    if prompt_image is not None:
      pipeline_id += "_prompt_image"
      
    is_use_lora = lora_finetune and lora_finetune != ""
    
    # load pipeline
    if self.pipeline_id != pipeline_id or self.is_use_lora != is_use_lora or self.is_low_vram != is_low_vram:
      # clear caches
      print("clean caches")
      self.pipeline = None
      self.pipeline_id = None
      torch.cuda.empty_cache()
      gc.collect()
      
      # select and load pipeline
      if model_name == SDXL_MODEL_NAME and prompt_image is not None:
        print("load StableDiffusionXLAdapterPipeline")
        adapter = df.T2IAdapter.from_pretrained("TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")
        euler_a = df.EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", cache_dir=MODELS_PATH, use_safetensors=True)
        vae = df.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=MODELS_PATH, use_safetensors=True)
        self.pipeline = df.StableDiffusionXLAdapterPipeline.from_pretrained(model_name, adapter=adapter, scheduler=euler_a, vae=vae, torch_dtype=torch.float16, variant="fp16", cache_dir=MODELS_PATH, use_safetensors=True)
      elif model_name == SDXL_MODEL_NAME and prompt_image is None:
        print("load StableDiffusionXLPipeline")
        euler_a = df.EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", cache_dir=MODELS_PATH, use_safetensors=True)
        vae = df.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=MODELS_PATH, use_safetensors=True)
        self.pipeline = df.StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16")
      else:
        print("load DiffusionPipeline")
        self.pipeline = df.DiffusionPipeline.from_pretrained(model_name, cache_dir=MODELS_PATH, use_safetensors=True, torch_dtype=torch.float16, variant="fp16")

			# optimize pipeline for low vram
      self.pipeline.to("cuda")
      if is_low_vram:
        print("optimize for low vram")
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_xformers_memory_efficient_attention()
      
      # save some settings
      self.pipeline_id = pipeline_id
      self.is_use_lora = is_use_lora
      self.is_low_vram = is_low_vram
    
    # handle seed
    if seed is None or seed <= 0: seed = None
    current_seed = seed or torch.randint(0, int(1e5), size=(1, 1))[0].item()
    gn = torch.Generator().manual_seed(int(current_seed))
    
    # handle lora finetune
    if is_use_lora:
      print("load_lora_weights")
      self.pipeline.load_lora_weights(LORA_FINETUNES_PATH, weight_name=lora_finetune)
    else:
      print("unload_lora_weights")
      self.pipeline.unload_lora_weights()
    
    # run pipeline
    if prompt_image is not None:
      # run SDXL pipeline with adapter
      pidinet = cn.pidi.PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
      image = pidinet(prompt_image, detect_resolution=width, image_resolution=width, apply_filter=True)
      result_images = self.pipeline(prompt=prompt, image=image, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=prompt_guidance, generator=gn, width=int(width), height=int(height), adapter_conditioning_scale=prompt_image_guidance, num_images_per_prompt=batch).images
    else:
      # run Generic pipeline
      result_images = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=prompt_guidance, generator=gn, width=int(width), height=int(height), num_images_per_prompt=batch).images
    
    # save and preview generation results
    for image in result_images: save_result_img(image, current_seed)
    print("done")
    return [result_images, current_seed]

main = Main()

with gr.Blocks() as demo:
  with gr.Tab("Генерация картинок"):
    with gr.Row():
      with gr.Column(scale=1):
        prompt_image = gr.Image(label="Промпт картинка", interactive=True)
        model_name = gr.Dropdown(label="Название модели", value=SUGGESTED_MODELS[0], choices=SUGGESTED_MODELS, allow_custom_value=True, filterable=True, interactive=True)
        lora_finetune = gr.Dropdown(label="Lora файнтюн", value="", choices=[""], allow_custom_value=False, filterable=False, interactive=True)
        prompt = gr.Textbox(max_lines=1, value=default_prompt, label="Положительный промпт")
        negative_prompt = gr.Textbox(max_lines=1, value=default_negative_prompt, label="Негативный промпт")
        with gr.Row():
          steps = gr.Slider(label="Шаги", minimum=0, maximum=150, step=1, value=50, interactive=True)
          prompt_guidance = gr.Slider(label="Внимание к промпту", minimum=0, maximum=10, step=0.5, value=8, interactive=True)
          prompt_image_guidance = gr.Slider(label="Внимание к промпт картинке", minimum=0, maximum=1, step=0.1, value=0.9, interactive=True)
        with gr.Row():
          width = gr.Dropdown(label="Ширина", value=1024, choices=WIDTH_CHOICES, allow_custom_value=True, filterable=True, interactive=True)
          height = gr.Dropdown(label="Высота", value=1024, choices=HEIGHT_CHOICES, allow_custom_value=True, filterable=True, interactive=True)
          batch = gr.Number(label="Количество результатов", value=1, interactive=True)
          seed = gr.Number(label="Seed", value=0, interactive=True)
        with gr.Row():
          is_low_vram = gr.Checkbox(label="Низкая видеомапять", value=True)
        generate_btn = gr.Button("Сгенерировать")
      with gr.Column(scale=1):
        result_images = gr.Gallery(label="Результаты", interactive=False, height=1024*0.75)
        result_seed = gr.Textbox(max_lines=1, label="Seed", interactive=False, show_copy_button=True)

  lora_finetune.focus(fn=update_lora_finetune_choices, outputs=[lora_finetune], show_progress=False)
  prompt_image.upload(fn=allow_only_sdxl_model, outputs=[model_name], show_progress=False)
  prompt_image.clear(fn=allow_all_models, outputs=[model_name], show_progress=False)
  generate_btn.click(fn=main.generate, inputs=[prompt_image, prompt_image_guidance, model_name, prompt, negative_prompt, width, height, steps, prompt_guidance, batch, seed, lora_finetune, is_low_vram], outputs=[result_images, result_seed])

if __name__ == "__main__":
  demo.launch(show_api=False, inbrowser=True, show_error=True)