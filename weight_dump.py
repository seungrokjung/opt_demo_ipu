import torch
import argparse 
from transformers import set_seed
from transformers import OPTForCausalLM
import os 

import gc 
import smooth

import numpy as np 

set_seed(123)

def save_weights(weights_dir, model_org, modle_name):
    if model_org == "local_dir":
        model = OPTForCausalLM.from_pretrained(model_name)
    else:
        model = OPTForCausalLM.from_pretrained(args.model_path)
    if (args.quant_mode == "none"):
        pass
    else:
        if (args.quant_mode == "smooth"):
            if model_name == "chatopt_1.3b_gpt4only":
                act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "opt-1.3b.pt") #exception: hard-coded
            else:
                act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%model_name)
            smooth.smooth_lm(model, act_scales, 0.5)
            print(f"SmoothQuant enabled ...")
        torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True )
        torch.save(model, "./quantized_%s.pth"%model_name)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.modules.linear.Linear):
            weight_bias = module._packed_params._weight_bias()
            weight_q = torch.int_repr(
                weight_bias[0]).numpy().astype( np.int8)
            weight_scale = weight_bias[0].q_scale()
            
            fname = weights_dir + "/" + name 

            if weight_bias[1] is not None:
                bias = weight_bias[1].detach().numpy()
                print(f"{name} {module._get_name()} {weight_q.shape} {bias.shape} ")
                count += bias.shape[0]
                np.savez(fname, weight_q=weight_q, weight_scale=weight_scale, bias=bias)
            else:
                print(f"{name} {module._get_name()} {weight_q.shape} None ")
                bias = None
                np.savez(fname, weight_q=weight_q, weight_scale=weight_scale)
            
            count += weight_q.shape[0] * weight_q.shape[1]
    print(f"Num of params: {count/(1024*1024)}MB")

def read_weights(weights_dir, model_org):
    for path, directories, files in os.walk(weights_dir):
        for i, file_name in enumerate(files):
            file_name = path + "/" + file_name 
            npzfile = np.load(file_name)
            weight_q = npzfile['weight_q']
            weight_scale = npzfile['weight_scale']
            
            if 'bias' in npzfile.files:
                bias = npzfile['bias']
                print(f"{file_name} {weight_q.shape} {bias.shape} {weight_q.min()} {weight_q.max()}")
            else:
                bias = None
                print(f"{file_name} {weight_q.shape} None ")
            

if __name__ == "__main__":
    """
    Description:
    1. Load OPT model
    2. Perform Smooth quant
    3. Perform PTDQ
    4. Create weights directory
    5. Dump all integer weights, floating point scale and floating point bias to npz file
    6. Each npz file is the hierarchical name of the layer
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", help="Different OPT model sizes", type=str, default="facebook/opt-1.3b", choices=["facebook/opt-1.3b", "local_dir/chatopt_1.3b_gpt4only"])
    parser.add_argument('--quant_mode', help="Quantization mode - none, smoothquant or pytorch dynamic-quant", type=str, default="smooth", choices=["none", "dyn", "smooth"])
    parser.add_argument('--action', help="save to npz or read from npz", type=str, default="save", choices=["save", "read"])
    args = parser.parse_args()
    print(f"{args}")

    model_org = args.model_path.split("/")[0]
    model_name = args.model_path.split("/")[1]

    weights_dir = "./weights_%s"%model_name
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    if args.action == "save":
        save_weights(weights_dir, model_org, model_name)
    else:
        read_weights(weights_dir)