import multiprocessing

# 09:14

def greet(name):
  return f"Hello {name}!"

def on_closed():
  print("quit")
  quit()

def gradio_process_worker():
  import gradio as gr
  import torch
  print(torch.rand(5, 3))
  print("ready")

def webview_process_worker():
  import webview
  main_window = webview.create_window("Woah dude!", html="<h1>Loading...<h1>")
  main_window.events.closed += on_closed
  webview.start()

def main():
  gradio_process = multiprocessing.Process(target=gradio_process_worker)
  webview_process = multiprocessing.Process(target=webview_process_worker)

  gradio_process.start()
  webview_process.start()
  webview_process.join()
  gradio_process.join()

if __name__ == "__main__": main()