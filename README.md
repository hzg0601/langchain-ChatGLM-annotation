2023-06-28新增的特性和修复的bug包括：

1. 增加自动下载文件的脚本auto_recursive_download.py，可以自动断点重连下载文件；
2. 增加批量修改huggingface_hub下载的文件名的shell脚本batch_rename_real_files.sh.(但有时不能成功，与shell脚本所处的环境有关，暂不清楚具体原因)
3. 增加对guanaco-65-q5量化模型的支持。
   1. 但推理速度巨慢。。。。。原因可能是模型默认加载到内存里，虽然支持部分层offload到显存中，但由于cpu-gpu通讯等原因拖慢了推理速度
4. 增加对guanaco-33b的支持。
   1. 加载模型需要约67-70G显存/内存，建议开启--load-in-8bit参数，推理速度也很慢，但实测效果较chatglm,moss,baichuan等模型要好不少。
   2. 默认的多GPU部署方案负载极不均衡，有待进一步优化。

2023-06-27新增的特性和修复的bug包括：

1. 修改了moss, baichuan,chatyuan,bloom模型的元prompt；
2. 新增了moss_llm.py的chat函数；
3. 重写了moss_llm.py的gerateAnser方法，基于chat函数进行问答；
4. 调整了生成文本的默认参数。

**基于webui.py测试了baichuan-7b，实测模型问答表现提高了不少**

2023-06-26新增的特性和修复的bug，包括：

1. 支持chatglm2-6b的多卡部署。
   **基于cli_demo.py进行了测试**

截止2023-06-20新增的特征和修复的bug,包括：
   **在cli_demo.py中测试**

1. 未本地提前下载模型的情况下，自动断点重连下载并加载模型.

   国内加载模型经常出现ConnectionError,所以需要多次下载多次调用AutoClass.from_pretrained加载模型，直至成功或超过 `max_try`次
2. 增加了llama-cpp模型的支持。
   llama-cpp模型需要在loader.py中调用llama-cpp-python中的Llama类，并使设定的参数兼容Llama类的generate方法，基于不
   同时期的ggml 格式手动下载llama-cpp-python版本，重新InvalidScoreLogitsProcessor类，转换输入input_ids的格式等操作。
   目前为兼容llama_llm.py脚本进行了参数阉割，后续似应考虑重新一个类。
   **通过cli_deom.py在lmsys/vicuna-13b-delta-v1.1/ggml-vicuna-13b-1.1-q5上，及llama-cpp-python v0.1.63上进行了测试。**
3. 修复了moss_llm.py的bug。
   在api.py\webui.py等下游调用脚本中，会在初始化后调用.set_history_len(LLM_HISTORY_LEN)，但moss_llm.py对该方法的定义
   与chatglm_llm.py不一致，导致调用失败。疑似能解决moss启动失败的问题[不能本地加载moss模型吗？ #652](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/652)，[加载本地moss模型报错Can&#39;t instantiate abstract class MOSSLLM with abstract methods _history_len #578](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/578)，[[BUG] moss模型本地加载报错 #577](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/577)，[[BUG] moss模型无法加载 #356](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/356)，[[BUG] moss_llm没有实现 #447](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/447)
   该错误在调用chatyuan模型时发现。
   **通过api.py脚本进行了测试**
4. 修复了对chatyuan模型的支持 [[BUG] config 使用 chatyuan 无法启动 #604](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/604) [chatyuan无法使用 #475](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/475) [ChatYuan-large-v2模型加载失败 #300](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/300) [[BUG] 使用chatyuan模型时，对话Error，has no attribute &#39;stream_chat&#39; #282](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/282) [[BUG] 使用ChatYuan-V2模型无法流式输出，会报错 #277](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/277) [增加Dockerfile 和ClueAI/ChatYuan-large-v2 模型的支持 #152](https://github.com/imClumsyPanda/langchain-ChatGLM/pull/152)
   chatyuan模型需要调用在loader.py中基于AutoModel进行调用，并使用MOSSLLM类作为provides，但当前脚本虽然在
   model_config.py中定义了模型配置字典，但并未进行正式适配，本次提交进行了正式支持。
   **通过api.py进行了测试**
5. 增加了对bloom模型的支持.[[FEATURE] 未来可增加Bloom系列模型吗？根据甲骨易的测试，这系列中文评测效果不错 #346](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/346)
   bloom模型对中文和代码的支持优于llama等模型，可作为候选模型。
   **通过api.py对bloom-3b进行了测试，bloomz-7b由于没有资源而没有测试**
6. 增加了对baichuan-7b模型的支持 [[FEATURE] 是否能够支持baichuan模型 #668](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/668)。
   **通过api.py进行了测试**
7. 修复多GPU部署的bug，多GPU部署时，默认的device-map为chatglm的device-map，针对新模型几乎必然会报错
   改为使用accelerate的infer_auto_device_map。
   该错误由于调用bloom等模型时发现。
   **通过api.py和cli_demo.py脚本进行了测试**
8. 增加对openai支持，参考openai的调用方式，重写了fastchat的模型配置字段，使支持openai的调用。
   **没有openai的key,因此没有测试**
9. 在install.md文档中，增加了对调用llama-cpp模型的说明。
   要调用llama-cpp模型需要的手动配置的内容较多，需要额外进行说明
10. 支持在多卡情况下也可以自定义部署在一个GPU设备。[设置了torch.cuda.set_device(1)，不起作用 #681](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/681) [请问如何设置用哪一张GPU？ #693](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/693)
    在某些情况下，如机器被多人使用，其中一个卡利用率较高，而另一些利用率较低，此时应该将设备指定在空闲的卡上，而不应自动进行多卡并行，但在现有版本中，load.py中_load_model没有考虑这种情况，clear_torch_cache函数也没有考虑这种情况。本次提交中，支持在model_config.py中设置llm_device的为类似“cuda:1”的方式自定义设备。
11. 优化了[FEATURE][[FEATURE] bing搜索问答有流式的API么？ #617](https://github.com/imClumsyPanda/langchain-ChatGLM/issues/617)，增加了api.py的stream_chat_bing接口；
12. 优化了api.py的stream_chat的接口,更改为在请求体中选择knowledge_base_id，从而无需两次指定knowledge_base_id；
13. 优化了cli_demo.py的逻辑：
    1 增加了输入提示；
    2 支持多个文件输入；
    3 支持文件输入错误有再次输入，而非直接因报错而退出；
    4 支持部分文件因路径错误而未加载成功后，可选再次加载。
