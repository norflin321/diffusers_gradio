import os

PWD = os.getcwd()
LORA_FINETUNES_PATH = f"{PWD}/lora_finetunes"
RESULTS_PATH = f"{PWD}/results"
MODELS_PATH = f"{PWD}/models"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXLL = "SDXL Lightning (fast)"
SUGGESTED_MODELS = [SDXL, SDXLL]
WIDTH_CHOICES = [1024, 768, 512]
HEIGHT_CHOICES = [1024, 768, 512]
DEF_PROMPT = "a magic creature in style of sks, 3D, blender, perfect, 4k graphics, highly detailed, cute, pretty"
DEF_N_PROMPT = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"