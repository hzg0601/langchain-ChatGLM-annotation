from multiprocessing import Process, Queue
import multiprocessing as mp
import subprocess
import sys
import os
from xml.etree.ElementPath import prepare_child

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.model_config import llm_model_dict, LLM_MODEL, LLM_DEVICE, LOG_PATH, logger
from configs.server_config import (WEBUI_SERVER, API_SERVER, OPEN_CROSS_DOMAIN, FSCHAT_CONTROLLER, FSCHAT_MODEL_WORKERS,
                                FSCHAT_OPENAI_API, fschat_controller_address, fschat_model_worker_address,)
from server.utils import MakeFastAPIOffline, FastAPI
import argparse
from typing import Tuple, List


def set_httpx_timeout(timeout=60.0):
    import httpx
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout


def create_controller_app(
        dispatch_method: str,
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller

    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    return app


def create_model_worker_app(**kwargs) -> Tuple[argparse.ArgumentParser, FastAPI]:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id
    import argparse
    import threading
    import fastchat.serve.model_worker

    # workaround to make program exit with Ctrl+c
    # it should be deleted after pr is merged by fastchat
    def _new_init_heart_beat(self):
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=fastchat.serve.model_worker.heart_beat_worker, args=(self,), daemon=True,
        )
        self.heart_beat_thread.start()
    ModelWorker.init_heart_beat = _new_init_heart_beat

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    # default args. should be deleted after pr is merged by fastchat
    args.gpus = None
    args.max_gpu_memory = "20GiB"
    args.load_8bit = False
    args.cpu_offloading = None
    args.gptq_ckpt = None
    args.gptq_wbits = 16
    args.gptq_groupsize = -1
    args.gptq_act_order = False
    args.awq_ckpt = None
    args.awq_wbits = 16
    args.awq_groupsize = -1
    args.num_gpus = 1
    args.model_names = []
    args.conv_template = None
    args.limit_worker_concurrency = 5
    args.stream_interval = 2
    args.no_register = False

    for k, v in kwargs.items():
        setattr(args, k, v)

    if args.gpus:
        if args.num_gpus is None:
            args.num_gpus = len(args.gpus.split(','))
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )

    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
    )

    sys.modules["fastchat.serve.model_worker"].worker = worker
    sys.modules["fastchat.serve.model_worker"].args = args
    sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
    
    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({LLM_MODEL})"
    return app


def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys

    MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app


def _set_app_seq(app: FastAPI, q: Queue, run_seq: int):
    # 如果run_seq == 1,则定义FastAPI的on_event事件starup(可选shutdown),
    # 定义函数为async def on_startup:函数内容为先定义httpx的各种timeout时间
    # 然后将run_seq放进队列mp.Queue
    if run_seq == 1:
        @app.on_event("startup")
        async def on_startup():
            set_httpx_timeout()
            q.put(run_seq)
    # 如果run_seq不为1，则同样定义FastAPI的on_event事件startup
    # 定义函数为async def on_startup:函数内容为先定义httpx的各种timeout时间
    # 然后调用Queue.get从队列取值，如果取值不为run_seq-1，将值再放入队列，
    # 从而使该进程一直阻塞，直至run_seq-1顺序的进程被放入Queue中，从而q.get
    # 得到前序任务序号run_seq-1,然后将run_seq放入Queue中。
    elif run_seq > 1:
        @app.on_event("startup")
        async def on_startup():
            set_httpx_timeout()
            while True:
                no = q.get()
                if no != run_seq - 1:
                    q.put(no)
                else:
                    break
            q.put(run_seq)


def run_controller(q: Queue, run_seq: int = 1):
    import uvicorn

    app = create_controller_app(FSCHAT_CONTROLLER.get("dispatch_method"))
    _set_app_seq(app, q, run_seq)

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]
    uvicorn.run(app, host=host, port=port)


def run_model_worker(
    model_name: str = LLM_MODEL,
    controller_address: str = "",
    q: Queue = None,
    run_seq: int = 2,
):
    import uvicorn

    kwargs = FSCHAT_MODEL_WORKERS[LLM_MODEL].copy()
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    model_path = llm_model_dict[model_name].get("local_model_path", "")
    kwargs["model_path"] = model_path
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address()

    app = create_model_worker_app(**kwargs)
    _set_app_seq(app, q, run_seq)

    uvicorn.run(app, host=host, port=port)


def run_openai_api(q: Queue, run_seq: int = 3):
    import uvicorn

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(controller_addr) # todo: not support keys yet.
    _set_app_seq(app, q, run_seq)

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    uvicorn.run(app, host=host, port=port)


def run_api_server(q: Queue, run_seq: int = 4):
    from server.api import create_app
    import uvicorn

    app = create_app()
    _set_app_seq(app, q, run_seq)

    host = API_SERVER["host"]
    port = API_SERVER["port"]

    uvicorn.run(app, host=host, port=port)


def run_webui(q: Queue, run_seq: int = 5):
    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    while True:
        no = q.get()
        if no != run_seq - 1:
            q.put(no)
        else:
            break
    q.put(run_seq)
    p = subprocess.Popen(["streamlit", "run", "webui.py",
                    "--server.address", host,
                    "--server.port", str(port)])
    p.wait()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="run fastchat's controller/model_worker/openai_api servers, run api.py and webui.py",
        dest="all",
        default=True
    )
    parser.add_argument(
        "-o",
        "--openai-api",
        action="store_true",
        help="run fastchat controller/openai_api servers",
        dest="openai_api",
    )
    parser.add_argument(
        "-m",
        "--model-worker",
        action="store_true",
        help="run fastchat model_worker server with specified model name. specify --model-name if not using default LLM_MODEL",
        dest="model_worker",
    )
    parser.add_argument(
        "-n"
        "--model-name",
        type=str,
        default=LLM_MODEL,
        help="specify model name for model worker.",
        dest="model_name",
    )
    parser.add_argument(
        "-c"
        "--controller",
        type=str,
        help="specify controller address the worker is registered to. default is server_config.FSCHAT_CONTROLLER",
        dest="controller_address",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="run api.py server",
        dest="api",
    )
    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="run webui.py server",
        dest="webui",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method("spawn")
    queue = Queue()
    args = parse_args()
    if args.all:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.webui = True

    logger.info(f"正在启动服务：")
    logger.info(f"如需查看 llm_api 日志，请前往 {LOG_PATH}")

    processes = {}

    if args.openai_api:
        # 用Process定义一个子进程process,
        # process的start方法开启子进程，子进程会调用每个函数的方法
        # 方法的_set_app_seq决定是否将len(procesess)+1其放入queue中
        # 初始时，len(procesess)+1=1，因此1被放入queue中，
        # 且controller开始执行
        # 同时其他进程开启，但由于_set_app_seq的机制，只有逻辑上第二个进程
        # 能开启，其他进程由于while True循环而无法开启，
        # 同时基于join的机制，主进程等待子进程结束后才会继续。
        process = Process(
            target=run_controller,
            name=f"controller({os.getpid()})",
            args=(queue, len(processes) + 1),
            daemon=True,
        )
        process.start()
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api({os.getpid()})",
            args=(queue, len(processes) + 1),
            daemon=True,
        )
        process.start()
        processes["openai_api"] = process

    if args.model_worker:
        process = Process(
            target=run_model_worker,
            name=f"model_worker({os.getpid()})",
            args=(args.model_name, args.controller_address, queue, len(processes) + 1),
            daemon=True,
        )
        process.start()
        processes["model_worker"] = process

    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server{os.getpid()})",
            args=(queue, len(processes) + 1),
            daemon=True,
        )
        process.start()
        processes["api"] = process

    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WEBUI Server{os.getpid()})",
            args=(queue,),
            daemon=True,
        )
        process.start()
        processes["webui"] = process

    try:
        if model_worker_process := processes.get("model_worker"):
            model_worker_process.join()
        for name, process in processes.items():
            if name != "model_worker":
                process.join()
    except:
        if model_worker_process := processes.get("model_worker"):
            model_worker_process.terminate()
        for name, process in processes.items():
            if name != "model_worker":
                process.terminate()

# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm2-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
