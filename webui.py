import gradio as gr
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import os

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()
# Each flagged sample (both the input and output data) is logged to a CSV file with headers on the machine running the gradio app
# 将flagged样例，包括输入和输出写入CSV的logger
flag_csv_logger = gr.CSVLogger()


def get_answer(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    """问答的后台函数"""
    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [
                    f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
            vs_path):
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库测试":
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_conent=chunk_conent,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], ""
    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


def init_model():
    """初始化模型的函数"""
    args = parser.parse_args()

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2, use_lora, top_k,
                 history):
    """重新初始化模型"""
    try:
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history,\
                gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。\n\n"
                                 "5. 本界面用于知识入库及知识匹配相关参数设定，但当前版本中，"
                                 "本界面中修改的参数并不会直接修改对话界面中参数，仍需前往`configs/model_config.py`修改后生效。"
                                 "相关参数将在后续版本中支持本界面直接修改。")


def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)


# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id, "vector_store"))
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(KB_ROOT_PATH, vs_id, "content"),
                                                                         vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())

def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = local_doc_qa.delete_file_from_vector_store(vs_path=vs_path,
                                                        filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = local_doc_qa.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files)>0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中无已上传文件，请先上传文件后，再开始提问。"
    logger.info(",".join(files_to_delete)+vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉langchain-ChatGLM WebUI🎉
👍 [https://github.com/imClumsyPanda/langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""欢迎使用 langchain-ChatGLM Web UI！

请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

知识库问答模式，选择知识库名称后，即可开始问答，当前知识库{default_vs}，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。

知识库暂不支持文件删除，该功能将在后续版本中推出。
"""

# 初始化消息
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)
# gr.Blocks(theme:Theme | str | None =None,analytics_enabled: bool | None =None,mode:str='blocks',title:str="Gradio",css:str|None=None)
# theme, 可以是内建的“soft","default","monochrome","glass"；gr.themes下的Theme类；或者HF HUB里的主题，如"gradio/monochrome"，monochrome单色
# analytics_enabled，是否允许搜集信息
# mode 当前Blocks的别名，默认为blocks
# title,在浏览器窗口中打开时显示的选项卡标题
# 应用于当前Blocks的自定义 css 或自定义 css 文件的路径

with gr.Blocks(css=block_css, theme=gr.themes.Soft(**default_theme_args,),title="chatglm-6b-webui-hzg") as demo:
    # 特殊的隐藏组件，用于存储同一用户运行演示时的会话状态。当用户刷新页面时，State 变量的值被清除。
    vs_path, file_status, model_status = gr.State(
        os.path.join(KB_ROOT_PATH, get_vs_list()[0], "vector_store") if len(get_vs_list()) > 1 else ""), gr.State(""), gr.State(
        model_status)
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
                                label="请选择使用模式",
                                value="知识库问答", )
                knowledge_set = gr.Accordion("知识库设定", visible=False)
                vs_setting = gr.Accordion("配置知识库")
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    vs_delete = gr.Button("删除本知识库", visible=False)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("删除文件"):
                            files_to_delete = gr.CheckboxGroup(choices=[],
                                                             label="请从知识库已有文件中选择要删除的文件",
                                                             interactive=True)
                            delete_file_button = gr.Button("从知识库中删除选中文件")
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    vs_delete.click(fn=delete_vs,
                                    inputs=[select_vs, chatbot],
                                    outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, files_to_delete, vs_delete])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot, files_to_delete], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot, files_to_delete], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode],
                                 [chatbot, query])
                    delete_file_button.click(delete_file,
                                             show_progress=True,
                                             inputs=[select_vs, files_to_delete, chatbot],
                                             outputs=[files_to_delete, chatbot])
    with gr.Tab("知识库测试 Beta"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["知识库测试"],  # "知识库问答",
                                label="请选择使用模式",
                                value="知识库测试",
                                visible=False)
                knowledge_set = gr.Accordion("知识库设定", visible=True)
                vs_setting = gr.Accordion("配置知识库", visible=True)
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with knowledge_set:
                    score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                label="知识相关度 Score 阈值，分值越低匹配度越高",
                                                precision=0,
                                                interactive=True)
                    vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                    label="获取知识库内容条数", interactive=True)
                    chunk_conent = gr.Checkbox(value=False,
                                               label="是否启用上下文关联",
                                               interactive=True)
                    chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                            label="匹配单段内容的连接上下文后最大长度",
                                            interactive=True, visible=False)
                    chunk_conent.change(fn=change_chunk_conent,
                                        inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
                                        outputs=[chunk_sizes, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs_test = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加单条内容或文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("添加单条内容"):
                            one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                            one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                            one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                   interactive=True)
                            load_conent_button = gr.Button("添加内容并加载知识库")
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs_test)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs_test, vs_name, vs_add, file2vs, chatbot])
                    select_vs_test.change(fn=change_vs_name_input,
                                     inputs=[select_vs_test, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs_test, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot], )
                    load_conent_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, one_title, sentence_size, chatbot,
                                                     one_conent, one_content_segmentation],
                                             outputs=[vs_path, files, chatbot], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, score_threshold, vector_search_top_k, chunk_conent,
                                  chunk_sizes],
                                 [chatbot, query])
    with gr.Tab("模型配置"):
        llm_model = gr.Radio(llm_model_dict_list,
                             label="LLM 模型",
                             value=LLM_MODEL,
                             interactive=True)
        no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
                                      label="加载本地模型",
                                      interactive=True)

        llm_history_len = gr.Slider(0, 10,
                                    value=LLM_HISTORY_LEN,
                                    step=1,
                                    label="LLM 对话轮数",
                                    interactive=True)
        use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                     label="使用p-tuning-v2微调过的模型",
                                     interactive=True)
        use_lora = gr.Checkbox(USE_LORA,
                               label="使用lora微调的权重",
                               interactive=True)
        embedding_model = gr.Radio(embedding_model_dict_list,
                                   label="Embedding 模型",
                                   value=EMBEDDING_MODEL,
                                   interactive=True)
        top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                          label="向量匹配 top k", interactive=True)
        load_model_button = gr.Button("重新加载模型")
        load_model_button.click(reinit_model, show_progress=True,
                                inputs=[llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2,
                                        use_lora, top_k, chatbot], outputs=chatbot)
        load_knowlege_button = gr.Button("重新构建知识库")
        load_knowlege_button.click(reinit_vector_store, show_progress=True,
                                   inputs=[select_vs, chatbot], outputs=chatbot)
    # load()
    # 出于兼容性考虑，.load即是一种类方法也是一种实例方法，但类方法和实例方法的实现不同
    # 类方法，用于从HF Space repo中加载demo，并创建一个block实例并返回

    # 实例方法，用于展示浏览器中加载demo后立即运行的事件
    # fn: 包装接口的函数，通常是机器学习模型的预测函数，如果函数有输入参数，在input里指定
    #   应返回单个值或一个元组，每个值对应输出的一个分量，即refresh_vs_list的值会在outputs
    #   定义的组件中显示。
    # inputs: fn的输入，默认为None
    # outputs: fn的返回值，默认为None
    # api_name: 如不为None, 将api文档中对外暴露端点
    # scroll_to_output: 如果为真，将在完成时滚动到输出组件
    # show_progress: 如果为真，将在挂起时显示进度动画
    # queue: 如果为 True，将把请求放在队列中;
    # batch: 如果为真，则该函数应该处理一批输入，这意味着它应该接受每个参数的输入值列表。 列表的长度应该相等（并且最大长度为“max_batch_size”）。
    #       然后该函数*需要*返回一个列表元组（即使只有 1 个输出组件），元组中的每个列表对应一个输出组件。
    # max_batch_size: int=4,如果从队列中调用，则要一起批处理的最大输入数（仅当 batch=True 时相关）
    # preprocess: 如果为 False，则在运行“fn”之前不会对组件数据进行预处理（例如，如果使用“Image”组件调用此方法，则将其保留为 base64 字符串）。
    # postprocess: 如果为 False，则在将“fn”输出返回给浏览器之前不会运行组件数据的后处理。
    # every: 指定多少秒运行一次事件，必须启用队列。
    # name:  the name of the model (e.g. "gpt2" or "facebook/bart-base") or space (e.g. "flax-community/spanish-gpt2"), 
    #        can include the `src` as prefix (e.g. "models/facebook/bart-base")
    # src:  模型的来源：`models` 或 `spaces`（如果在 `name` 中作为前缀提供来源，则留空）
    # api_key:  optional access token for loading private Hugging Face Hub models or spaces.
    # alias: 模型的别名
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs, select_vs_test],
        queue=True,
        show_progress=False,
    )

# queue(concurrency_count,status_update_rate,api_open,max_size)
# 通过创建队列来控制处理请求的速率。这将允许您设置一次要处理的请求数，并让用户知道他们在队列中的位置。
# concurrency_count,将同时处理来自队列的请求的工作线程数。增加这个数字会增加处理请求的速度，但也会增加队列的内存使用量。
# status_update_rate,如果为“auto”，Queue 将在作业完成时向所有客户端发送状态估计。否则，Queue 将定期发送此参数设置为秒数的状态。
# api_open,如果为 True，后端的 REST 路由将打开，允许直接向这些端点发出的请求跳过队列。
# max_size,队列在任何给定时刻存储的最大事件数。如果队列已满，则不会添加新事件，并且用户会收到一条消息，说明队列已满。如果没有，队列大小将是无限的。

#launch(inline,inbrowser,share,debug,max_threads,auth,auth_message,
# prevent_thread_lock,show_error,sever_name,server_port,show_tips,height,width,
# favicon_path,ssl_keyfile,ssl_certfile,ssl_keyfile_password,ssl_verify,quiet,show_api,
# allowed_paths,blocked_paths,root_paths,app_kwargs) 
# 启动一个web服务器
# inline:bool|None=None,是否在 iframe 中内联显示在界面中。在 python 笔记本中默认为 True；否则为假。
# inbrowser: bool=False,是否在默认浏览器的新选项卡中自动启动界面。
# share: 是否为界面创建可公开共享的链接。 创建一个 SSH 隧道，使您的 UI 可以从任何地方访问。 
#        如果未提供，则每次默认设置为 False，但在 Google Colab 中运行时除外。 当本地主机不可访问时（例如 Google Colab），不支持设置 share=False。
# debug: 如果为True，则阻塞主线程运行。
# max_threads: int=40, Gradio 应用程序可以并行生成的最大总线程数。 
#               默认继承自 starlette 库（当前为 40）。 无论队列是否启用都适用。 
#               但如果启用排队，则此参数将增加到至少为队列的 concurrency_count。
# auth: 如果提供，访问界面所需的用户名和密码（或用户名-密码元组列表）。还可以提供接受用户名和密码并在有效登录时返回 True 的功能。
# auth_message:str,如果提供，则在登录页面上提供 HTML 消息
# prevent_thread_lock: 如果为 True，该接口将在服务器运行时阻塞主线程。
# show_tips: if True, will occasionally show tips about new Gradio features
# favicon_path: 图标的路径
# quiet: If True, suppresses most print statements.
# show_api: 如果为真，则在应用程序的页脚中显示 api 文档。默认为真。如果启用队列，则 .queue() 的 api_open 参数将确定是否显示 api 文档，与 show_api 的值无关。
# allowed_path: list|None=None 允许 gradio 服务的完整文件路径或父目录的列表（除了包含 gradio python 文件的目录）。
#               必须是绝对路径。 警告：如果您提供目录，则您应用的所有用户都可以访问这些目录或其子目录中的任何文件。
# blocked_paths: 不允许 gradio 服务的完整文件路径或父目录列表（即不允许您的应用程序的用户访问）。 必须是绝对路径。 
#               警告：默认情况下优先于 `allowed_paths` 和 Gradio 公开的所有其他目录。
# root_paths: 应用程序的根路径（或“挂载点”），如果它不是从域的根（“/”）提供的。 通常在应用程序位于将请求转发给应用程序的反向代理后面时使用。 
#               例如，如果应用程序在“https://example.com/myapp”提供服务，则“root_path”应设置为“/myapp”。
# app_kwargs: 作为参数键和参数值的字典传递给底层 FastAPI 应用程序的附加关键字参数。例如，`{"docs_url": "/docs"}`
(demo
 .queue(concurrency_count=3)
 .launch(server_name='10.20.33.13',
         server_port=7860,
         show_api=True,
         share=True,
         inbrowser=False))

# .integrate(comet_ml,wandb,mlflow),一种与其他库集成的万能方法。此方法应在 launch() 之后运行
# comet_ml,如果提供了 comet_ml Experiment 对象，将与实验集成并出现在 Comet 仪表板上
# wandb,如果提供了 wandb 模块，将与其集成并出现在 WandB 仪表板上
# mlflow,如果提供了 mlflow 模块，将与实验集成并出现在 ML Flow 仪表板上
