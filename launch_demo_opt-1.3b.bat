start python -W ignore::UserWarning -m fastchat.serve.controller 2>&1
start python -W ignore::UserWarning -m fastchat.serve.model_worker --model-path facebook/opt-1.3b               --model-file quantized_opt-1.3b.pth 2>&1
python -W ignore::UserWarning -m fastchat.serve.gradio_web_server 