"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, OPTForCausalLM,  set_seed
import torch
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL

from utils import Utils
import gc 
import smooth
import os
import qlinear 

set_seed(123)
logger_use = False
if logger_use == True:
    from fastchat.utils import (build_logger, server_error_msg, pretty_print_semaphore)

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
if logger_use == True:
    logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

def load_model(model_path, model_file, num_gpus):
    model_org =  model_path.split("/")[0]
    #model_name = model_path.split("/")[1]
    model_name = "opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b") #exception: hard-coded
    print("Load INT8 model...")
    model = torch.load(model_file)
    model.eval()

    node_args = ()
    node_kwargs = {}
    print("Deploy smooth quant...")
    Utils.replace_node( model, 
                        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                        qlinear.QLinear, 
                        node_args, node_kwargs 
                      )
    collected = gc.collect()

    if num_gpus == 1:
        model.cuda("cuda:0")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048 # opt

    return tokenizer, model, context_len

class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_file, num_gpus
                 ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        #self.model_name = model_path.split("/")[-1]
        self.model_name = "opt-1.3b"

        if logger_use == True:
            logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.context_len = load_model(model_path, model_file, num_gpus)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        if logger_use == True:
            logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        if logger_use == True:
            logger.info(f"Send heart beat. Models: {[self.model_name]}. " f"Semaphore: {pretty_print_semaphore(model_semaphore)}. " f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                if logger_use == True:
                    logger.error(f"heart beat error: {e}")
                pass
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            # Fixed
            #return args.limit_model_concurrency - model_semaphore._value + len(model_semaphore._waiters)
            return args.limit_model_concurrency 

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0: # initial tkn gen
                out = model(torch.as_tensor([input_ids]), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1)
                out = model(input_ids=torch.as_tensor([[token]]), use_cache=True, attention_mask=attention_mask, past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            #print(output_ids)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                #output = tokenizer.decode(output_ids, skip_special_tokens=True)
                output = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break

        del past_key_values

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://127.0.0.1:21002")
    parser.add_argument("--controller-address", type=str, default="http://127.0.0.1:21005")
    parser.add_argument("--model-path", type=str, default="facebook/opt-1.3b", choices=["facebook/opt-1.3b", "local_dir/chatopt_1.3b_gpt4only", "local_dir/amd-hardcoded"])
    parser.add_argument("--model-file", type=str, default="quantized_opt-1.3b.pth", choices=["quantized_opt-1.3b.pth", "quantized_chatopt_1.3b_gpt4only.pth", "quantized_opt1.3b_merged_cnn-daily-0.3_gpt4-wo-orca-0822-clean97k-amd-hardcoded_continue-bingchat-amd.pth"])
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    if logger_use == True:
        logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_file,
                         args.num_gpus
                         )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")