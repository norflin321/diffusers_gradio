import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL
import os
import gc

# @TODO:
# 1. использовать генеирк StableDiffusion и без кастомных vae and scheduler (use_safetensors=True, cache_dir=f"{PWD}/models")
# 2. протестировать работу в собранной exe
# 3. сделать 2 числовых инпута под width and height
# 4. подобрать другие модели которые возможно лучше будут работать на слабом железе
# 5. lora_weights

PWD = os.getcwd()
MODELS = {
  "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0"
}
RESOLUTIONS = ["1024x1024", "768x768", "512x512"]

def save_result_img(img, name):
  if not os.path.exists(f"{PWD}/results"): os.makedirs(f"{PWD}/results")
  img.save(f"{PWD}/results/{name}.png")

class Main:
  pipeline: StableDiffusionXLPipeline = None
  pipeline_name: str = None

  def generate(self, model_name, prompt, negative_prompt, resolution, steps, guidance, seed):
    if self.pipeline_name != model_name:
      # clean pipeline
      if self.pipeline != None:
        self.pipeline = None
        self.pipeline_name = None
        gc.collect()
        torch.cuda.empty_cache()
      
      # create pipeline
      self.pipeline_name = model_name
      euler_a = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
      vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
      # self.pipeline = StableDiffusionXLPipeline.from_pretrained(MODELS[model_name], use_safetensors=True, variant="fp16", cache_dir=f"{PWD}/models", torch_dtype=torch.float16).to("cuda")
      self.pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16")
      self.pipeline.to("cuda")
      self.pipeline.enable_model_cpu_offload()
      self.pipeline.enable_xformers_memory_efficient_attention()
    
    # handle seed
    if seed <= 0: seed = None
    current_seed = seed or torch.randint(0, int(1e5), size=(1, 1))[0].item()
    gn = torch.Generator().manual_seed(int(current_seed))
    
    # run pipeline to generate an image
    [width, height] = resolution.split("x", 2)
    print([width, height])
    result_img = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gn, width=int(width), height=int(height)).images[0]
    # save_result_img(result_img, f"img_{current_seed}")
    print("done")
    return result_img

main = Main()

with gr.Blocks() as demo:
  with gr.Tab("Text to Img"):
    with gr.Row():
      with gr.Column(scale=1):
        model_select = gr.Dropdown(label="Model", value="SDXL Base 1.0", choices=MODELS.keys(), allow_custom_value=False, filterable=False, interactive=True)
        prompt = gr.Textbox(max_lines=1, label="Prompt")
        negative_prompt = gr.Textbox(max_lines=1, label="Negative Prompt")
        resolution = gr.Dropdown(label="Resolution", value=RESOLUTIONS[0], choices=RESOLUTIONS, allow_custom_value=False, filterable=False, interactive=True)
        steps = gr.Slider(label="Steps", minimum=0, maximum=100, step=1, value=50, interactive=True)
        guidance = gr.Slider(label="Guidance", minimum=0, maximum=10, step=0.5, value=8, interactive=True)
        seed = gr.Number(label="Seed", value=0, interactive=True)
        generate_btn = gr.Button("Generate")
      with gr.Column(scale=1):
        result_img = gr.Image(label="Result Img", interactive=False)

  generate_btn.click(fn=main.generate, inputs=[model_select, prompt, negative_prompt, resolution, steps, guidance, seed], outputs=[result_img])

if __name__ == "__main__":
  print(torch.cuda.is_available())
  demo.launch(show_api=False, inbrowser=True, show_error=True)