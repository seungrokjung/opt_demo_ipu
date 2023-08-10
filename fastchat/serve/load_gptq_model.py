import sys
from pathlib import Path
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import os
#sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
sys.path.append("/ROCM/MODEL/SD/SD/git_repos/oobabooga_GPTQ/GPTQ-for-LLaMa")
#sys.path.append("/ROCM/MODEL/SD/SD/git_repos/qwopqwop200_GPTQ/GPTQ-for-LLaMa/GPTQ-for-LLaMa")
from modelutils import find_layers
from quant import make_quant


def load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], device="cuda:0"):
    print(model)
    print(checkpoint)
    from transformers import LlamaConfig, LlamaForCausalLM 
    #config = LlamaConfig.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    #model = LlamaForCausalLM(config)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    #make_quant(model, layers, wbits, groupsize, faster=False)
    make_quant(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint, device), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


def load_quantized(model_name, wbits=4, groupsize=128):
    #model_name = model_name.replace('/', '_')
    path_to_model = Path(f'./{model_name}')
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None
    print(model_name)
    print(path_to_model)
    print(found_pts)
    print(found_safetensors)

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()
    device = "cuda:0"
    model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, device)

    return model
