# Python 标准库模块 argparse 可用于解析命令行参数~ 但由于 argparse 使用复杂，add_argument 方法参数众多。
# 为此，第三方库模块 click 应运而生，极大地改善了 argparse 的易用性。

#Click 第三方库由 Flask 的作者 Armin Ronacher 开发。Click 相较于 argparse 较好比 requests 相较于 urllib。

# 这里定义了一个一级组cli,定义了三个二级组:llm,embedding,start,
# start定义了api,cli,webui三个命令
import click

from api import api_start as api_start
from configs.model_config import llm_model_dict, embedding_model_dict

# @click.group 装饰器把方法装饰为可以拥有多个子命令的 Group 对象。
# 由 Group.add_command() 方法把 Command 对象关联到 Group 对象。 
# 也可以直接用 @Group.command 装饰方法，会自动把方法关联到该 Group 对象下。

# 即在使用python cli.py时，启动的入口函数为cli(),而其他属于该组的command，相当于cli()的子函数
#? 如果有多个组，该怎么绑定？-> 用组名.command的方式绑定，如llm.command
@click.group()
# Add a --version option which immediately prints the version number and exits the program.
@click.version_option(version='1.0.0')
# 将回调标记为想要接收当前上下文对象作为第一个参数。
@click.pass_context
def cli(ctx):
    pass


@cli.group()
def llm():
    pass

# name为命令的名字，默认情况下，就是函数名，但函数名的下划线需要替换为-
# 相当于为命令取别名
@llm.command(name="ls")
def llm_ls():
    for k in llm_model_dict.keys():
        print(k)

# A shortcut decorator for declaring and attaching a group to the group. 
# This takes the same arguments as group() 
# and immediately registers the created group with this group by calling add_command().
@cli.group()
def embedding():
    pass


@embedding.command(name="ls")
def embedding_ls():
    for k in embedding_model_dict.keys():
        print(k)


@cli.group()
def start():
    pass

# python cli.py start api --ip 127.0.0.1 --port 8001
@start.command(name="api", context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-i', '--ip', default='0.0.0.0', show_default=True, type=str, help='api_server listen address.')
@click.option('-p', '--port', default=7861, show_default=True, type=int, help='api_server listen port.')
def start_api(ip, port):
    api_start(host=ip, port=port)

# python cli.py start cli
@start.command(name="cli", context_settings=dict(help_option_names=['-h', '--help']))
def start_cli():
    import cli_demo
    cli_demo.main()

# python cli.py start webui
@start.command(name="webui", context_settings=dict(help_option_names=['-h', '--help']))
def start_webui():
    import webui


cli()
