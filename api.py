import argparse
import json
import os
# shutil库，它作为os模块的补充，提供了复制、移动、删除、压缩、解压等操作，这些 os 模块中一般是没有提供的。
# 但是需要注意的是：shutil 模块对压缩包的处理是调用 ZipFile 和 TarFile这两个模块来进行的。
import shutil
from typing import List, Optional
import urllib

import nltk
import pydantic #强制类型检查库
# 一个基于asyncio的ASGI web服务器
import uvicorn 
# fastapi Python网络框架，用于构建API，但不含任何服务器应用程序
# 自动生成交互式API接口文档
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing_extensions import Annotated
# Starlette 是一个轻量级的 ASGI 框架和工具包，特别适合用来构建高性能的 asyncio 服务
from starlette.responses import RedirectResponse

from chains.local_doc_qa import LocalDocQA
from configs.model_config import (VS_ROOT_PATH, UPLOAD_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# 定义fastapi中的复合参数，相当于C的结构体
class BaseResponse(BaseModel):
    # 如果不加self，表示是类的一个属性(可以通过"类名.变量名"的方式引用),加了self表示是类的实例的一个属性(可以通过"实例名.变量名"的方式引用)
    # pydantic.Filed,用于定义字段的各项检查限制，第一个为默认值
    # Used to provide extra information about a field, either for the model schema or complex validation
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


def get_folder_path(local_doc_id: str):
    return os.path.join(UPLOAD_ROOT_PATH, local_doc_id)


def get_vs_path(local_doc_id: str):
    return os.path.join(VS_ROOT_PATH, local_doc_id)


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(UPLOAD_ROOT_PATH, local_doc_id, doc_name)


async def upload_file(
        # fastapi.File用于修饰文件信息
        file: UploadFile = File(description="A single binary file"),
        # When you need to receive form fields instead of JSON, you can use Form.
        # Form 表单字段,例如带框的用户名和密码
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):  
    """
    用于上传一个文件
    file: 二进制文件
    knowledge_base_id: 知识库的id
    1. 先获取知识库ID的文件夹路径，如果不存在则创建；
    2. 读取文件内容；
    3. 构造文件路径，如果文件路径已存在且文件内容长度与文件路径下文件size一致，则返回基响应类
    4. 否则，向文件路径写入文件内容
    5. 构造向量数据库地址,初始化向量数据库，返回响应基类。
    """
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)
    # vs，vector_store向量数据库
    vs_path = get_vs_path(knowledge_base_id)
    # init_knowledge_vector_store方法的调用链：init_knowledge_vector_store->load_file->
    # 对于md，调用UnstructuredFileLoader即可, 
    #   #! 注意charset_normalizer模块需要较低的版本,测试在2.1.0上可用，
        # !但在3.1.0上会报partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
    # 对于其他格式，调用{UnstructuredPDF/Text/UnstructuredImage}FileLoader和ChineseTextSplitter
    # UnstructuredPaddle{Image/PDF}Loader调用paddleocr模块，可能会返回OSError: [Errno 101] Network is unreachable错误
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def upload_files(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    """
    用于上传一批文件
    """
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file in files:
        file_content = ''
        file_path = os.path.join(saved_path, file.filename)
        file_content = file.file.read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue
        with open(file_path, "ab+") as f:
            f.write(file_content)
        filelist.append(file_path)
    if filelist:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, get_vs_path(knowledge_base_id))
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
            return BaseResponse(code=200, msg=file_status)
    file_status = "文件未成功加载，请重新上传文件"
    return BaseResponse(code=500, msg=file_status)


async def list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    """
    列出知识库里的所有文档；
    1. 如果给定了知识库ID，则只列出该知识库对应文件件里的文件；
    2. 否则列出UPLOAD_ROOT_PATH路径下的所有文件；
    """
    if knowledge_base_id:
        local_doc_folder = get_folder_path(knowledge_base_id)
        if not os.path.exists(local_doc_folder):
            return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
        all_doc_names = [
            doc
            for doc in os.listdir(local_doc_folder)
            if os.path.isfile(os.path.join(local_doc_folder, doc))
        ]
        return ListDocsResponse(data=all_doc_names)
    else:
        if not os.path.exists(UPLOAD_ROOT_PATH):
            all_doc_ids = []
        else:
            all_doc_ids = [
                folder
                for folder in os.listdir(UPLOAD_ROOT_PATH)
                if os.path.isdir(os.path.join(UPLOAD_ROOT_PATH, folder))
            ]

        return ListDocsResponse(data=all_doc_ids)


async def delete_docs(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
        doc_name: Optional[str] = Query(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    """
    删除给定知识库里的文档；
    1. 若给定了文档名，则删除知识库内指定文档，更新剩余文档的向量数据库；
    2. 若没有给定文件名，则删除整个知识库的文件。
    """
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    if doc_name:
        doc_path = get_file_path(knowledge_base_id, doc_name)
        if os.path.exists(doc_path):
            os.remove(doc_path)

            # 删除上传的文件后重新生成知识库（FAISS）内的数据
            # 先列出知识库的所有剩余文件，若文件数量为0，则删除知识库内所有文件；
            # 否则初始化向量数据库
            remain_docs = await list_docs(knowledge_base_id)
            if len(remain_docs.data) == 0:
                shutil.rmtree(get_folder_path(knowledge_base_id), ignore_errors=True)
            else:
                local_doc_qa.init_knowledge_vector_store(
                    get_folder_path(knowledge_base_id), get_vs_path(knowledge_base_id)
                )
            
            return BaseResponse(code=200, msg=f"document {doc_name} delete success")
        else:
            return BaseResponse(code=1, msg=f"document {doc_name} not found")

    else:
        shutil.rmtree(get_folder_path(knowledge_base_id))
        return BaseResponse(code=200, msg=f"Knowledge Base {knowledge_base_id} delete success")


async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        # Body可以将单类型的参数成为 Request Body 的一部分，即从查询参数变成请求体参数
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    """
    基于本地知识库文档的chat.
    1. 首先构造本地知识库的向量数据库地址，如果地址不存在，则返回错误；
    2. 如果数据库存在，则调用local_doc_qa.get_knowledge_based_answer进行chat,
        该方法先根据问题搜索向量数据库，找到最相似的文档，将文档与问题组装成prompt，
        再调用LLM进行chat；
        该方法由yield控制的生成器，返回response, history
        response为一个字典，包括query,answer,source_documents；
    3. 返回query,answer,source_documents，history等信息

    """
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return ChatMessage(
            question=question,
            response=f"Knowledge base {knowledge_base_id} not found",
            history=history,
            source_documents=[],
        )
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=source_documents,
        )


async def bing_search_chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: Optional[List[List[str]]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    """
    基于bing搜索的结果进行聊天，返回query,answer,source_documents，history等信息
        1. 直接调用BingSearchAPIWrapper搜索问题，返回result_len个结果；
        2. 调用langchain.docstore.document.Document将返回的结果包装为Document实例；
        3. 将包装后的搜索结果与问题一起组装为prompt；
        4. 调用LLM生成答案，返回的source即搜索的结果；

    """
    for resp, history in local_doc_qa.get_search_result_based_answer(
            query=question, chat_history=history, streaming=True
    ):
        pass
    source_documents = [
        f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(resp["source_documents"])
    ]

    return ChatMessage(
        question=question,
        response=resp["result"],
        history=history,
        source_documents=source_documents,
    )

async def chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):  
    """
    基于LLM模型进行聊天
        直接调用LLM模型进行问答，因此不会返回source_document.
    """
    for answer_result in local_doc_qa.llm.generatorAnswer(prompt=question, history=history,
                                                          streaming=True):
        resp = answer_result.llm_output["answer"]
        history = answer_result.history
        pass

    return ChatMessage(
        question=question,
        response=resp,
        history=history,
        source_documents=[],
    )


async def stream_chat(websocket: WebSocket, knowledge_base_id: str):
    """
    基于知识库的流水式问答
        1. question, history, knowledge_base_id由websocket的receive_json方法传入
        2. 若存在知识库的向量数据库，将question经websocket.send_json发出
        3. 调用local_doc_qa.get_knowledge_based_answer进行知识问答，将结果result由websocket.send_text返回；
        4. 将question,turn,source_document由websocket.send_text返回


    """
    # websocket是一种网络通信协议，是 HTML5 开始提供的一种在单个 TCP 连接上进行全双工通讯的协议。
    # WebSocket用于在Web浏览器和服务器之间进行任意的双向数据传输的一种技术。WebSocket协议基于TCP协议实现，
    # 包含初始的握手过程，以及后续的多次数据帧双向传输过程。
    # 其目的是在WebSocket应用和WebSocket服务器进行频繁双向通信时，
    # 可以使服务器避免打开多个HTTP连接进行工作来节约资源，提高了工作效率和资源利用率。
    # WebSocket目前支持两种统一资源标志符ws和wss，类似于HTTP和HTTPS。
    # 浏览器发出webSocket的连线请求，服务器发出响应，这个过程称为握手,握手的过程只需要一次，就可以实现持久连接。

    # accept,send,receive,receive_*,send_*,close, 
    # accept先调用receive，然后调用send({"type": "websocket.accept", "subprotocol": subprotocol, "headers": headers})
    await websocket.accept()
    # 轮次
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question, history, knowledge_base_id = input_json["question"], input_json["history"], input_json["knowledge_base_id"]
        vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)

        if not os.path.exists(vs_path):
            await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
            await websocket.close()
            return

        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def document():
    return RedirectResponse(url="/docs")


def api_start(host, port):
    # 调用FastAPI构建API
    global app
    global local_doc_qa
    # 初始化大模型
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    app = FastAPI()
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        # Web的middleware是指在Web应用程序中用于处理HTTP请求和响应的中间件组件。
        # 中间件是一种软件模块，它可以在处理请求和响应之前或之后执行某些操作，例如身份验证、日志记录、缓存、压缩等等。

        # 在Web应用程序中，中间件通常以管道的形式组合在一起，
        # 每个中间件都可以对请求或响应进行处理并将其传递给下一个中间件。
        # 这种模式使得开发人员可以轻松地添加、删除或替换中间件，以实现特定的功能或满足特定的需求。
        
        # CORSMiddleware类，它是用于处理跨域资源共享（CORS）的中间件。
        # CORS是一种Web浏览器的安全机制，用于限制在不同域名之间共享资源的能力。
        # 例如，如果一个网站的JavaScript代码试图从一个不同的域名请求数据，浏览器就会阻止该请求。
        # 为了允许跨域请求，服务器需要设置特定的HTTP头。
        # CORSMiddleware类就是用来帮助开发人员在Starlette应用程序中实现CORS功能的。

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # allow_origins选项指定了允许的来源，*表示允许所有来源。
        # allow_credentials选项指定了是否允许发送身份验证凭据，True表示允许。
        # allow_methods选项指定了允许的HTTP方法，*表示允许所有方法。
        # allow_headers选项指定了允许的HTTP头，*表示允许所有头。
    # 功能类似于装饰器 ，如下文相当于
    # 用@app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}") 装饰stream_chat函数。  
    app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")(stream_chat)
    # post(): 用于定义HTTP POST请求的API端点。POST请求通常用于向服务器提交数据，例如表单数据或JSON数据。
    # 当客户端发送POST请求时，FastAPI会将请求的路径映射到该方法，并将请求体解析为Python对象。
    # 开发人员可以在方法的参数中接受解析后的对象，并返回一个响应。

    # get(): 用于定义HTTP GET请求的API端点。GET请求通常用于从服务器获取数据，例如HTML页面或JSON数据。
    # 当客户端发送GET请求时，FastAPI会将请求的路径映射到该方法，并解析查询字符串参数。
    # 开发人员可以在方法的参数中接受解析后的参数，并返回一个响应。

    # websocket(): 用于定义WebSocket连接的API端点。WebSocket是一种实时通信协议，它允许在客户端和服务器之间进行双向通信。
    # 当客户端发送WebSocket连接请求时，FastAPI会将请求的路径映射到该方法。
    # 开发人员可以在方法中接受WebSocket连接对象，并使用异步生成器来处理来自客户端的消息。
    app.get("/", response_model=BaseResponse)(document)

    app.post("/chat", response_model=ChatMessage)(chat)

    app.post("/local_doc_qa/upload_file", response_model=BaseResponse)(upload_file)
    app.post("/local_doc_qa/upload_files", response_model=BaseResponse)(upload_files)
    app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    app.post("/local_doc_qa/bing_search_chat", response_model=ChatMessage)(bing_search_chat)
    app.get("/local_doc_qa/list_files", response_model=ListDocsResponse)(list_docs)
    # 用于定义HTTP DELETE请求的API端点。DELETE请求通常用于从服务器删除数据。
    app.delete("/local_doc_qa/delete_file", response_model=BaseResponse)(delete_docs)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    api_start(args.host, args.port)
