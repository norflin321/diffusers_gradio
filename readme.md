```
venv\Scripts\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U git+https://github.com/huggingface/diffusers.git controlnet_aux==0.0.7
pip install transformers accelerate protobuf sentencepiece safetensors mediapipe invisible_watermark
pip install gradio
pip install pyinstaller

pyi-makespec --collect-data=gradio_client --collect-data=gradio --onefile main.py
a = Analysis(
    ...
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    },
)

pyinstaller main.spec - build exe

gradio main.py - start with hot reload
```
https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
https://huggingface.co/docs/diffusers/using-diffusers/sdxl


https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local (select Window 10 or 11)
nvcc --version

- pip install -U git+https://github.com/huggingface/diffusers.git controlnet_aux==0.0.7
- pip install transformers accelerate protobuf sentencepiece safetensors mediapipe invisible_watermark
- pip install gradio pyinstaller
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- pip install -U xformers --index-url https://download.pytorch.org/whl/cu121