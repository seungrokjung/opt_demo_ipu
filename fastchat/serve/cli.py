"""
Usage:
python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
"""
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import conv_templates, SeparatorStyle


#@torch.inference_mode()
def generate_stream(tokenizer, model, params, device,
                    context_len=2048, stream_interval=2):
    """Adapted from fastchat/serve/model_worker.py::generate_stream"""

    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]


    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=args.min_length,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    if 0: 
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids], device=device))
                   # torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device)
                out = model(input_ids=torch.as_tensor([[token]], device=device),
                            #use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False


    if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))
        #output = tokenizer.decode(output_ids, skip_special_tokens=True)
        output = tokenizer.decode([el.item() for el in generated_ids[0]])
        pos = output.rfind(stop_str, l_prompt)
        if pos != -1:
            output = output[:pos]
            stopped = True
        yield output

    #if stopped:
    #    break

    #del past_key_values



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--model", type=str, default="facebook/opt-350m")
    parser.add_argument("--load", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--conv-template", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wbits", type=int, default = 0)
    parser.add_argument("--groupsize", type=int, default = 0)
    args = parser.parse_args()

    model_name = args.model_name
    num_gpus = args.num_gpus

    # Model
    if args.device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif args.device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if args.wbits > 0:
        from fastchat.serve.load_gptq_model import load_quantized, load_quant

        print("Loading GPTQ quantized model...")

        #model = load_quantized(model_name, args.wbits, args.groupsize)
        model = load_quant(args.model, args.load, args.wbits, args.groupsize, args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

    #if args.device == "cuda" and num_gpus == 1:
    #    model.cuda()

    print(model)
    model.to("cuda")
    model = torch.compile(model)
    input_ids = tokenizer.encode("hihiih", return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=20,
            max_length=40,
        )

    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
    # Chat
    conv = conv_templates[args.conv_template].copy()
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        print(f"{conv.roles[1]}: ", end="", flush=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        print(input_ids)
        pre = 0
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            max_length=2000,
        )
        for outputs in generate_stream(tokenizer, model, params, args.device):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        print(" ".join(outputs[pre:]), flush=True)

        conv.messages[-1][-1] = " ".join(outputs)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
