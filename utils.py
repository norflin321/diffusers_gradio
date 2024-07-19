import datetime
import os
import const as const
import gradio as gr

def save_result_img(img, seed):
  time_now = datetime.datetime.now().strftime("%y.%m.%d_%H-%M-%S")
  img.save(f"{const.RESULTS_PATH}/{time_now}_{seed}.png")
  
def allow_only_sdxl_model():
  return gr.update(choices=[const.SDXL_MODEL_NAME], value=const.SDXL_MODEL_NAME, interactive=True)

def allow_all_models():
  return gr.update(choices=const.SUGGESTED_MODELS, interactive=True)

def update_lora_finetune_choices():
  choices = [f for f in os.listdir(const.LORA_FINETUNES_PATH) if os.path.isfile(os.path.join(const.LORA_FINETUNES_PATH, f))]
  choices.insert(0, "")
  return gr.update(choices=choices, interactive=True, value=None)