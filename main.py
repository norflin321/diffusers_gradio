import multiprocessing as mp

is_main_ready = mp.Value('b', False)
is_main_alive = mp.Value('b', True)

def quit_main():
  is_main_alive.value = False
  quit()

def webview_process_worker(is_main_ready):
  import webview

  main_window = webview.create_window("Woah dude!", html="<h1>Loading...<h1>")
  main_window.events.closed += quit_main

  def listen_main():
    is_loading_shown = True
    while True:
      if not is_main_alive.value: quit() # quit this process if main quited

      # wait untill main process is ready, then hide loading screen
      if is_main_ready.value and is_loading_shown:
        print("hide loading", is_main_ready.value, is_loading_shown)
        is_loading_shown = False

  webview.start(listen_main)

def main():
  try:
    # create child process to run webview inside of it
    webview_process = mp.Process(target=webview_process_worker, args=(is_main_ready,))
    webview_process.start()

    # start gradio
    import gradio as gr
    import torch
    print(torch.rand(5, 3))

    is_main_ready.value = True # tell webview process that it can hide loading screen

    webview_process.join()
  except Exception as e:
    quit_main()

if __name__ == "__main__":
  try: main()
  except KeyboardInterrupt: quit_main()