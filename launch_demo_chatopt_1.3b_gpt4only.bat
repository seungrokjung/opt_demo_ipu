start python -W ignore::UserWarning -m fastchat.serve.controller 2>&1
start python -W ignore::UserWarning -m fastchat.serve.model_worker --model-path local_dir/chatopt_1.3b_gpt4only --model-file quantized_chatopt_1.3b_gpt4only.pth 2>&1
python -W ignore::UserWarning -m fastchat.serve.gradio_web_server 