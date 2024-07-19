import datetime
import os
import const as const
import gradio as gr

def save_imgs(imgs, seed):
  time_now = datetime.datetime.now().strftime("%y.%m.%d_%H-%M-%S")
  for img in imgs: img.save(f"{const.RESULTS_DIR}/{time_now}_{seed}.png")
  
def allow_only_sdxl_model():
  return gr.update(choices=[const.SDXL], value=const.SDXL, interactive=True)

def allow_all_models():
  return gr.update(choices=const.SUGGESTED_MODELS, interactive=True)

def update_lora_finetune_choices():
  choices = [f for f in os.listdir(const.LORA_DIR) if os.path.isfile(os.path.join(const.LORA_DIR, f))]
  choices.insert(0, "")
  return gr.update(choices=choices, interactive=True, value=None)

def log(*args):
  out = f"__[LOG]__:"
  for idx, arg in enumerate(args):
    out += f", {arg}" if idx != 0 else f" {arg}"
  print(out)