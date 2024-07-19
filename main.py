import gradio as gr
import os
import utils as utils
import const as const
from General import General

# ensure some required directories exists
if not os.path.exists(const.LORA_FINETUNES_PATH): os.makedirs(const.LORA_FINETUNES_PATH)
if not os.path.exists(const.RESULTS_PATH): os.makedirs(const.RESULTS_PATH)
if not os.path.exists(const.MODELS_PATH): os.makedirs(const.MODELS_PATH)

# set directory where diffusers will download models and other files
os.environ["DIFFUSERS_CACHE"] = const.MODELS_PATH
os.environ["HF_HOME"] = const.MODELS_PATH

general = General()

# GRADIO UI
with gr.Blocks() as demo:
  with gr.Tab("Универсальная генерация картинок"):
    with gr.Row():
      with gr.Column(scale=1):
        img = gr.Image(label="Промпт картинка", interactive=True)
        model = gr.Dropdown(label="Название модели", value=const.SUGGESTED_MODELS[0], choices=const.SUGGESTED_MODELS, allow_custom_value=True, filterable=True, interactive=True, info="Любая из этого списка https://huggingface.co/models?library=diffusers")
        lora = gr.Dropdown(label="Lora файнтюн", value="", choices=[""], allow_custom_value=False, filterable=False, interactive=True)
        txt = gr.Textbox(max_lines=1, value=const.DEF_PROMPT, label="Положительный промпт")
        n_txt = gr.Textbox(max_lines=1, value=const.DEF_N_PROMPT, label="Негативный промпт")
        with gr.Row():
          steps = gr.Slider(label="Шаги", minimum=0, maximum=150, step=1, value=50, interactive=True)
          txt_guid = gr.Slider(label="Внимание к промпту", minimum=0, maximum=10, step=0.5, value=8, interactive=True)
          img_guid = gr.Slider(label="Внимание к промпт картинке", minimum=0, maximum=1, step=0.1, value=0.9, interactive=True)
        with gr.Row():
          w = gr.Dropdown(label="Ширина", value=1024, choices=const.WIDTH_CHOICES, allow_custom_value=True, filterable=True, interactive=True)
          h = gr.Dropdown(label="Высота", value=1024, choices=const.HEIGHT_CHOICES, allow_custom_value=True, filterable=True, interactive=True)
          batch = gr.Number(label="Количество результатов", value=1, interactive=True)
          seed = gr.Number(label="Seed", value=0, interactive=True)
        with gr.Row():
          low_vram = gr.Checkbox(label="Оптимизировать расход видеопамяти за счет снижения скорости или качества", value=True)
        generate_btn = gr.Button("Сгенерировать")
      with gr.Column(scale=1):
        res_imgs = gr.Gallery(label="Результаты", interactive=False, height=1024*0.75)
        res_seed = gr.Textbox(max_lines=1, label="Seed", interactive=False, show_copy_button=True)
  
  lora.focus(fn=utils.update_lora_finetune_choices, outputs=[lora], show_progress=False)
  generate_btn.click(fn=general.generate, inputs=[img, model, lora, txt, n_txt, steps, txt_guid, img_guid, w, h, batch, seed, low_vram], outputs=[res_imgs, res_seed])

if __name__ == "__main__":
  demo.launch(show_api=False, inbrowser=True, show_error=True)