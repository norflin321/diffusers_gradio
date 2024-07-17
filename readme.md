```
venv\Scripts\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyinstaller
pip install gradio
pip uninstall numpy
pip install "numpy<2.0"
pip install asyncio

pyi-makespec --collect-data=gradio_client --collect-data=gradio --onefile main.py
a = Analysis(
    ...
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    },
)

pyinstaller main.spec
```