```
preferably create fresh venv

pip install gradio
pip install pyinstaller
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip uninstall numpy
pip install "numpy<2.0"
pip install pythonnet
pip install pywebview
pip install asyncio

pyi-makespec --collect-data=gradio_client --collect-data=gradio --onefile --windowed main.py
a = Analysis(
    ...
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    },
)

pyinstaller main.spec
```