import gradio as gr

def greet(name):
    return f"Hello {name}!"

def main():
  ui = gr.Interface(fn=greet, inputs="text", outputs="text", allow_flagging="never")
  ui.launch()

if __name__ == "__main__":
  main()