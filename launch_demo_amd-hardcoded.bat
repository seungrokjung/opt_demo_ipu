start python -W ignore::UserWarning -m fastchat.serve.controller 2>&1
start python -W ignore::UserWarning -m fastchat.serve.model_worker --model-path local_dir/amd-hardcoded --model-file quantized_opt1.3b_merged_cnn-daily-0.3_gpt4-wo-orca-0822-clean97k-amd-hardcoded_continue-bingchat-amd.pth  2>&1
python -W ignore::UserWarning -m fastchat.serve.gradio_web_server 