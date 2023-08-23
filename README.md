# OPT demo on Ryzen AI IPU
OPT demo user interface for serving and evaluating GPT-based Chatbot. 
The demo system consists of three parts. 
- Web ux/ui: Provides user interface with configurable parameters, such as temperature for token decoding sampling & max output token length
- Model worker: Launches, OPT models in quantizable format so that IPU can offload Matmul kernel operations. 
- Controller: Links Model worker and Web ux/ui

## News
- Web ux/ui sample

<a ><img src="assets/web_uxui.jpg" width="100%"></a>

## Install

### Launch anaconda prompt as an administrator and follow these steps
1. IPU compatible matmul kernel registeration

Install git in the beginning
```bash
conda install -c anaconda git
```
Clone the IPU transformers environment repo (https://gitenterprise.xilinx.com/VitisAI/transformers/tree/release_2) 
Follow instructions in the README 
Install the packages to the following path:
C:\Users\Transformers\transformers

2. Clone release_2_0823 branch of this repository (https://github.com/seungrokjung/opt_demo_ipu.git) to the following folder:
C:\Users\Transformers\demo

```bash
cd C:\Users\Transformers\demo
git clone https://github.com/seungrokjung/opt_demo_ipu.git -b release_2_0823
cd C:\Users\Transformers\demo\opt_demo_ipu
```

3. Install Package
```bash
pip install -r requirements.txt
```

## Open-source model: facebook/opt-1.3b:

Quantize the model:
```bash
python weight_dump.py --action save --model_path facebook/opt-1.3b
```

You can see the quantized torch model in the current directory.
```bash
quantized_opt-1.3b.pth
```

Launch a demo script and wait a 2-3 minutes until the quantized model is loaded to ddr. 
This script lanches 1) controller, 2) model worker, and 3) web ux/ui separately.
```bash
launch_demo_opt-1.3b
```

## AMD's fine-tuned model: chatopt_1.3b_gpt4only:

Copy the model to the current directory
cp -rf chatopt_1.3b_gpt4only .
After you copy the model, the model folder in the current path should look like this:
<a ><img src="assets/directory_structure.jpg" width="100%"></a>

Quantize the model:
```bash
python weight_dump.py --action save --model_path local_dir/chatopt_1.3b_gpt4only
```

You can see the quantized torch model in the current directory.
```bash
quantized_chatopt_1.3b_gpt4only.pth
```

Launch a demo script and wait a 2-3 minutes until the quantized model is loaded to ddr. 
This script lanches 1) controller, 2) model worker, and 3) web ux/ui separately.
```bash
launch_demo_chatopt_1.3b_gpt4only
```

## Check the demo status

Once the demo is ready, the environment will look like this (The ERROR message in controller window is negligible):
<a ><img src="assets/demo_setup.jpg" width="100%"></a>


## Accessing the chatbot 
Open a web browswer and navigate to "localhost:1001"


## Modification

## Restrictions

##
