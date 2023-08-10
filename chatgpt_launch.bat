start python -W ignore::UserWarning -m fastchat.serve.controller 2>&1
start python -W ignore::UserWarning -m fastchat.serve.model_worker 2>&1
python -W ignore::UserWarning -m fastchat.serve.gradio_web_server 