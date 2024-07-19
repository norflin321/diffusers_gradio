import os

PWD = os.getcwd()
LORA_FINETUNES_PATH = f"{PWD}/lora_finetunes"
RESULTS_PATH = f"{PWD}/results"
MODELS_PATH = f"{PWD}/models"
SDXL_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SUGGESTED_MODELS = [SDXL_MODEL_NAME]
WIDTH_CHOICES = [1024, 768, 512]
HEIGHT_CHOICES = [1024, 768, 512]
default_prompt = "a magic creature in style of sks, 3D, blender, perfect, 4k graphics, highly detailed, cute, pretty"
default_negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"