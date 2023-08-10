# OPT demo on Ryzen AI IPU
OPT demo user interfacet for serving and evaluating GPT-based Chatbot. The demo system consists of three parts. 
- Web ux/ui: Provides user interface with configurable parameters, such as temperature for token decoding sampling & max output token length
- Model worker: Launches, OPT models in quantizable format so that IPU can offload Matmul kernel operations. 
- Controller: Links Model worker and Web ux/ui

## News
- Web ux/ui sample

<a ><img src="assets/web_uxui.jpg" width="100%"></a>

## Contents
- [Install](#install)
- [Model Weights](#model-weights)
- [Inference with Command Line Interface](#inference-with-command-line-interface)
- [Serving with Web GUI](#serving-with-web-gui)
- [API](#api)
- [Evaluation](#evaluation)
- [Fine-tuning](#fine-tuning)
- [Citation](#citation)

## Install

### Launch anaconda prompt as an administrator and follow these steps
1. IPU compatible matmul kernel registeration

Install git in the beginning
```bash
conda install -c anaconda git
```

Clone the IPU transformers environment repo (https://gitenterprise.xilinx.com/VitisAI/transformers/tree/release_2) and follow instructions in the README 
Install the packages in the following path:
C:\Users\Transformers\transformers

2. Clone this repository (https://github.com/seungrokjung/opt_demo_ipu.git) to the following folder:
C:\Users\Transformers\demo\opt_demo_ipu

```bash
cd C:\Users\Transformers\demo
git clone https://github.com/seungrokjung/opt_demo_ipu.git
cd C:\Users\Transformers\demo\opt_demo_ipu
```

3. Install Package
```bash
pip install -r requirements.txt
```

4. Launch demo scripts and wait a few minutes until the model is preloaded & quantized. 
```bash
chatgpt_launch.bat
```

5. Once the demo is ready, the environment will look like this:
<a ><img src="assets/demo_setup.jpg" width="100%"></a>

6. Open a web browswer and navigate to "localhost:1001". Default username/password for the uxui are "amd/7890".
