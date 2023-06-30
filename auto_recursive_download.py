"""
可以同该脚本下载repo或文件，如果max_try设的很大仍不能成功，只能手动下载文件，然后用batch_create_soft_link.sh脚本建立软链接
"""
from huggingface_hub import hf_hub_download, snapshot_download
import os 
import shutil
import subprocess
import argparse
from typing import Union
def recursively_load_model(LoaderClass,
                           checkpoint,
                           config=None,
                           torch_dtype=None,
                           trust_remote_code=True, 
                           resume_download=True,
                           max_try=300,
                           kwargs=None):
    """国内加载模型经常出现ConnectionError,所以需要多次重复下载
        多次调用LoadClass.from_pretrained加载模型，直至成功或超过`max_try`次
    """
    try_turn = 0
    while True:
        try:
            
            if config is not None:
                print(config,checkpoint)
                print("*"*80)
                model = LoaderClass.from_pretrained(checkpoint,
                                             config=config,
                                             torch_dtype=torch_dtype,
                                             trust_remote_code=trust_remote_code,
                                             resume_download=resume_download) 
            else:
                print(checkpoint, kwargs)
                print("-"*80)
                model = LoaderClass.from_pretrained(checkpoint,**kwargs)
            return model
        except Exception as e:
            print("Download model failed, re-downloading...")
            try_turn += 1
            if try_turn > max_try:
                print("The number of retries exceeded `max_try`,maybe you are offline, please check the network.")
                raise ConnectionError
            else:
                continue    

def recursive_download_file(repo_id,
                            filename=None,
                            repo_type="model",
                            local_dir_use_symlinks=False,
                            cache_dir=None,
                            local_dir=None,
                            rename_files=False,
                            max_try=300):
    """
    自动短点重下指定文件直至成功
    repo_id: 仓库地址，如“TheBloke/guanaco-65B-GGML”
    filename: 指定文件名
    repo_type: model, dataset, space
    cache_dir:指定缓存地址
    local_dir: 指定本地存在地址
    local_dir_use_symlinks: 比较复杂，简言之是是否建立`local_dir`和`cache_dir`的软链接
    rename_files: 是否将下载的文件名改为真实文件名，若True，则使用默认地址，若str,则该参数需指向真实文件所在的地址；
                下载的huggingface文件通常会被重命名为一个随机文件名，然后用软链接的方式链接真实文件名，
                此参数运行将随机文件名改为真实文件名，
                #! 有时能成功有时不能成功，可能跟shell脚本所处的环境有关
    max_try: 最大重复下载次数，超过此值认为处于离线环境，抛出网络异常
    """
    try_turn = 0
    while True:
        try:
            
            if filename is not None:
                print("downloading",repo_id, filename)
                print("*"*80)
                file_dir = hf_hub_download(repo_id,
                                filename=filename,
                                repo_type=repo_type,
                                cache_dir=cache_dir,
                                local_dir=local_dir,
                                resume_download=True,
                                local_dir_use_symlinks=local_dir_use_symlinks
                                )

            else:
                print(repo_id)
                print("-"*80)
                file_dir = snapshot_download(repo_id,
                                repo_type=repo_type,
                                cache_dir=cache_dir,
                                local_dir=local_dir,
                                resume_download=True,
                                local_dir_use_symlinks=local_dir_use_symlinks
                                  )
            
            print("download file or repo succeed!")
            
            if rename_files: 
                rename_files = "/".join(file_dir.split("/")[:-1]) if isinstance(rename_files,bool) else rename_files

                copy_shell_name = os.path.join(rename_files,"batch_rename_real_files.sh")
                # 将目录下的batch_rename_real_files.sh移至真实文件所在文件夹下
                shutil.copyfile("./batch_rename_real_files.sh",copy_shell_name)
                current_path = os.getcwd()
                os.chdir(rename_files)
                # 实测用python执行shell脚本，其中的readlink -f link或realpath link命名不能获取
                # 真实地址，而是只能得到软连接的地址
                # todo 查查该怎么操作以及为什么会这样
                subprocess.run(["/bin/bash",copy_shell_name])
                os.remove(copy_shell_name)
            return file_dir
        except Exception as e:
            print("Download model failed, re-downloading...")
            try_turn += 1
            if try_turn > max_try:
                print("The number of retries exceeded `max_try`,maybe you are offline, please check the network.")
                raise ConnectionError
            else:
                continue 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="auto download from huggingface_hub",description="自动从huggingface hub下载文件")
    parser.add_argument("--repo-id",type=str,default="bigscience/tokenizer")
    parser.add_argument("--filename",type=str,default=None)
    parser.add_argument("--cache-dir",type=str,default=None)
    parser.add_argument('--repo-type',type=str,default="model")
    parser.add_argument('--local-dir',type=str,default=None)
    parser.add_argument('--local-dir-use-symlinks',type=bool, default=False)

    parser.add_argument("--rename-files",type=Union[bool,str,None],default=False,
                        help="""whether to rename the downloaded file, if True, use huggingface_hub's default directory; 
                        if str,input the directory of the soft links""")
    parser.add_argument("--max-try",type=int,default=3000)
    args = parser.parse_args()
    args_dict = vars(args)
    recursive_download_file(**args_dict)
