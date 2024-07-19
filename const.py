import os

PWD = os.getcwd()
LORA_DIR = f"{PWD}/lora_finetunes"
RESULTS_DIR = f"{PWD}/results"
CACHE_DIR = f"{PWD}/models"
SDXL = "SDXL"
SDXLL = "SDXL Lightning"
SUGGESTED_MODELS = [SDXL, SDXLL]
WIDTH_CHOICES = [1024, 768, 512]
HEIGHT_CHOICES = [1024, 768, 512]
DEF_PROMPT = "a magic creature in style of sks, 3D, blender, perfect, 4k graphics, highly detailed, cute, pretty"
DEF_N_PROMPT = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"