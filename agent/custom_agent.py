"""
CustomPromptTemplate根据agent_template定义模板实例prompt，该实例的.format方法迭代式给每一步的具体prompt
LLMChain根据llm模型和模板实例prompt定义链实例llm_chain, llm_chain核心方法是run,它返回的是调用generate方法输出prompt的结果，
    generate方法首先调用PromptTemplate实例prompt的format_prompt方法构造prompt实例列表和stop词，
    然后根据prompt实例列表和stop词调用llm.generate_prompt方法，得到LLMResult实例，即response
    然后调用create_output方法将generate返回的response.generations解析为str.
LLMSingleActionAgent根据llm_chain和OutputParser的实例构造一个agent，agent的核心是plan方法，
    plan方法根据prompt调用llm_chain给出的关于调用工具的信息，
    该信息是一个str,然后调用OutputParser.parse方法输出AgentFinish或AgentAction.
AgentExecutor.from_agent_and_tools根据agent和tools构造一个AgentExecutor实例，实例的核心方法是run,
    run方法调用的是agent_executor的_call方法，该方法在_should_continue(iterations, time_elapsed)循环中，
    不断调用_take_next_step，_take_next_step先调用agent.plan,agent.plan基于intermediate_steps调用llm_chain，llm_chain调用run,
    run调用generate方法，generate方法调用PromptTemplate实例prompt的format_prompt方法构造prompt实例列表和stop词,
    然后调用llm模型的.generate_prompt方法，得到LLMResult实例即response，将其解析为str,agent.plan然后基于str和OutputParser.parse方法
    输出AgentFinish或AgentAction,如果返回AgentFinish实例，则agent_executor的_call直接调用_return返回结果，
    如果返回的是AgentAction实例或AgentAction实例的列表，则agent_executor的_call针对每个action.tool得到Tool,
    调用tool.run执行action,得到observation,将observation、action拼接作为intermediate_steps列表，
    intermediate_steps列表进入下一轮_should_continue(iterations, time_elapsed)循环，直至结束循环；
    如果直至_should_continue(iterations, time_elapsed)循环停止，没有遇到返回值为AgentAction的实例，或者tool_return的返回值为None,
    则调用agent.return_stopped_response得到output,然后调用_return最终返回output的结果
故一个代理的构成包括：工具实例列表，CustomPromptTemplate实例，LLMChain（由llm模型和CustomPromptTemplate实例化）实例，
Agent实例（由llm_chain实例和OutputParser实例来实例化），AgentExecutor实例（由工具列表和agent实例来实例化）。

"""
from langchain.agents import Tool
# The BaseTool automatically infers the schema from the _run method’s signature.
# 自定义CustomTool时，仅需def _run 和 async def _arun即可
from langchain.tools import BaseTool
from langchain import PromptTemplate, LLMChain
from agent.custom_search import DeepSearch
from langchain.agents import BaseSingleActionAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from typing import List, Tuple, Any, Union, Optional, Type
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.base_language import BaseLanguageModel
import re

agent_template = """
你现在是一个{role}。这里是一些已知信息：
{related_content}
{background_information}
{question_guide}：{input}

{answer_format}
"""

# String prompt should expose the format method, returning a prompt
class CustomPromptTemplate(StringPromptTemplate):
    """自定义prompt模板，模板的目标是是提示llm根据中间步骤调用相应工具，调用工具是逐步进行的，即：
    如果format函数的输入参数中intermediate_steps为空，表示没有中间步骤了，或者直接回答问题，或者调用最终的工具DeepSearch进行搜索回答问题；
    如果format函数输入的intermediate_steps不为空，则表明存在中间步骤，中间步骤由两部分构成：action,observation,其中action即动作，
    observation为动作执行的结果，以动作执行的结果为最新的背景信息，组成最新的prompt,供下次llm_chain调用。
    上述过程不断进行，直至intermediate_steps为空。
    """
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        # 没有互联网查询信息
        if len(intermediate_steps) == 0:
            background_information = "\n"
            role = "傻瓜机器人"
            question_guide = "我现在有一个问题"
            answer_format = "如果你知道答案，请直接给出你的回答！如果你不知道答案，请你只回答\"DeepSearch('搜索词')\"，并将'搜索词'替换为你认为需要搜索的关键词，除此之外不要回答其他任何内容。\n\n下面请回答我上面提出的问题！"

        # 返回了背景信息
        else:
            # 根据 intermediate_steps 中的 AgentAction 拼装 background_information
            background_information = "\n\n你还有这些已知信息作为参考：\n\n"
            action, observation = intermediate_steps[0]
            background_information += f"{observation}\n"
            role = "聪明的 AI 助手"
            question_guide = "请根据这些已知信息回答我的问题"
            answer_format = ""

        kwargs["background_information"] = background_information
        kwargs["role"] = role
        kwargs["question_guide"] = question_guide
        kwargs["answer_format"] = answer_format
        return self.template.format(**kwargs) # 此处调用的是str类的.format方法，将对应的str添加到template里
# 没有用到的类
class CustomSearchTool(BaseTool):
    name: str = "DeepSearch"
    description: str = ""

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        return DeepSearch.search(query = query)

    async def _arun(self, query: str):
        raise NotImplementedError("DeepSearch does not support async")
# 没有用到的类
class CustomAgent(BaseSingleActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermedate_steps: List[Tuple[AgentAction, str]],
            **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        return AgentAction(tool="DeepSearch", tool_input=kwargs["input"], log="")

class CustomOutputParser(AgentOutputParser):
    """
    用于解析llm_chain根据template的实例prompt输出的结果的解析，
    如果工具调用链结束则返回AgentFinish,如果调用链未结束则返回AgentAction
    """
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # group1 = 调用函数名字
        # group2 = 传入参数
        match = re.match(r'^[\s\w]*(DeepSearch)\(([^\)]+)\)', llm_output, re.DOTALL)
        print(match)
        # 如果 llm 没有返回 DeepSearch() 则认为直接结束指令
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        # 否则的话都认为需要调用 Tool，match第一组为tool，第二组为tool的输入
        else:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class DeepAgent:
    """
    
    该类是通过 AgentExecutor.from_agent_and_tools定义agent_executor,需要定义agent,tools
    其中LLMSingleActionAgent定义agent, 需要定义output_parser,llm_chain;
        llm_chain需要实例化llm和prompt_template->CustomPromptTemplate；
        output_parser为AgentOutputParser子类的实例，需要定义parse方法，返回AgentAction实例或 AgentFinish实例；
        
    llm_chain是LLMChain的实例，需要定义llm模型实例和prompt模板（CustomPromptTemplate）实例

    DeepAgent类的逻辑是通过定义的query方法，调用agent_executor.run(agent_executor通过AgentExecutor.from_agent_and_tools定义)方法，
    run方法是Chain的方法，它接受一个位置参数，或关键字参数，run方法返回一个str,
    run方法调用的是agent_executor的_call方法，_call方法返回一个dict,
    _call方法的实现逻辑为：
        1. 首先构造tool_name和Tool的映射字典，然后构造tool_name到颜色的映射字典用于log打印
        2. 然后进入_should_continue(iterations, time_elapsed)循环：
            2.1 首先调用_take_next_step得到输出
                输出第一步是调用agent.plan得到返回值，如果返回值是AgentFinish实例，直接返回，
                如果返回值是action列表，针对每个action.tool和工具映射字典得到Tool,
                调用tool.run执行action,得到observation,将observation、action拼接，
                构造为一个列表返回；
            2.2 如果输出是AgentFinish实例，则直接调用_return取出next_step_output.return_values，
                它是一个字典,将intermediate_steps加入字典，返回结果
            2.3 如果输出不是AgentFinish实例而是AgentAction实例，
                但next_step_output只剩一步，则调用_get_tool_return得到tool_return
                然后调用_return返回结果，否则继续执行下去；
        3. 如果直至_should_continue(iterations, time_elapsed)循环停止，
           没有遇到返回值为AgentAction的实例，或者tool_return的返回值为None,
           则调用agent.return_stopped_response得到output,然后调用_return最终返回output的结果
    
    本类中，agent是通过LLMSingleActionAgent（是BaseSingleActionAgent的子类）实现的，它的plan方法是
    直接调用llm_chain.run方法得到输出(一个str，关于需要调用工具的信息->llm_chain能给出此类信息是根据prompt_template提示输出的),
    然后用OutputParser的parse方法解析输出的str，最终返回一个ActionFinish或ActionAgent实例。
    总之，如果llm_chain根据prompt_template的提示认为需要调用工具，则会返回关于工具的str,交由OutputParser.parse方法解析

    本脚本的OutputParser的parse方法逻辑为：
        如果 llm_chain.run方法得到输出（一个str，match("tool",str)的group1为tool,group2为tool_input）
            没有返回 "DeepSearch() "则认为直接结束指令，返回AgentFinish实例
        如果 llm_chain.run方法得到输出 返回了 DeepSearch() 则返回一个AgentAction实例
    故而_call调用的返回值next_step_output最多只剩一步，则调用_get_tool_return得到tool_return，然后调用_return返回结果

    LLMChain也是Chain的子类，它的run方法也是Chain的run方法，调用的是LLMChain的_call方法，而_call方法先调用generate方法，
    generate方法首先调用PromptTemplate实例prompt的format_prompt方法构造prompt实例列表和stop词，
    然后根据prompt实例列表和stop词调用llm.generate_prompt方法，得到LLMResult实例，即response
    然后调用create_output方法将generate返回的response.generations构造为一个字典列表，取出第一个字典，
    并取出self.output_key[0]的值(agent.return_values)，故 --Chain、LLMChain类的run方法最终返回的是一个str--

    LLMChain基于llm和prompt-template进行初始化，template给出了模型的action指令顺序，每次llm_chain.run都根据prompt_template
    和intermediate_steps(包括action,observation)输出一个template的具体prompt，
    该prompt包含了上次调用tool执行action的observation,以其为背景信息，不断执行
    intermediate_steps得到最新的template的prompt实例，直至没有中间步骤，最终回答问题（直接回答或调用最终工具，如本脚本的DeepSearch）。
    """
    tool_name: str = "DeepSearch"
    agent_executor: any
    tools: List[Tool]
    llm_chain: any

    def query(self, related_content: str = "", query: str = ""):
        tool_name = self.tool_name
        result = self.agent_executor.run(related_content=related_content, input=query ,tool_name=self.tool_name)
        return result

    def __init__(self, llm: BaseLanguageModel, **kwargs):
        # 1. 定义工具链，每个工具都以Tool.from_function的方式构造
        tools = [
                    Tool.from_function(
                        func=DeepSearch.search, # 其他例子，llm_math_chain.run,必须是执行动作的函数
                        name="DeepSearch",
                        description=""
                    )
                ]
        self.tools = tools
        tool_names = [tool.name for tool in tools]
        # 2. 定义输出的解析器
        output_parser = CustomOutputParser()
        # 3. 根据agent的模板和工具链，实例化提示模板
        prompt = CustomPromptTemplate(template=agent_template,
                                      tools=tools,
                                      input_variables=["related_content","tool_name", "input", "intermediate_steps"])
        # 4. 根据llm和提示模板定义链
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.llm_chain = llm_chain
        # 5. 根据链、输出解析器、停止词、工具名调用LLMSingleActionAgent实例化agent
        # 也可以通过调用from langchain.agents import AgentType, initialize_agent
        # initializer_agent(tools,llm,agent=AgentType.`type`)定义一个agent_executor
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        # 6. 根据实例化的agent，工具链调用AgentExecutor.from_agent_and_tools定义agent_executor
        # 实际使用时，通过调用agent_executor.run(related_content="", input="", tool_name="tool_name")

        # AgentExecutor是Chain的子类，agent_executor.run调用Chain的run方法，其接受一个位置参数，或关键字参数
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        self.agent_executor = agent_executor

