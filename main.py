import asyncio
import gradio as gr
import torch

def greet(a, b):
  return "hello"

async def main():
  print(torch.rand(5, 3))
  ui = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
    allow_flagging="never",
  )
  ui.launch(show_api=False, inbrowser=True, show_error=True, server_name="0.0.0.0")

if __name__ == "__main__":
  try: asyncio.run(main())
  except Exception: quit()