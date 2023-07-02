# 在windows下载的文件，如果直接用SCP等工具转移到Linux系统，工具会将软连接指向的文件替代软连接本身上传到linux中
# 从而使得linux上存在两份内容相同的文件名不同的文件，本脚本用于删除具有随机文件名的文件，并重新建立软连接

# 多用户的系统中，非root用户的bash并不是/usr/bin/bash,因此直接执行shell脚本有时候会失败
# 此时需要将终端的环境变量PATH或HOME在shell脚本中显式地export
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/huangzhiguo/.local/bin:/home/huangzhiguo/bin
root_dir=/home/huangzhiguo/.cache/huggingface/hub/

hub_dirs=(models--huggyllama--llama-13b)
# hub_dir, 如 models--moka-ai-m3e-base
for hub_dir in $hub_dirs
do
    # 排除shell脚本文件
    if [ $hub_dir != $(basename "$0$") ];then
        # blobs, refs, snapshots
        echo the hub_dir is:---- $hub_dir-----------
        tar_dir_blobs=$root_dir$hub_dir"/blobs/"

        echo the target_dir_to_blobs is: $tar_dir_blobs
        # 首先删除blobs下的所有文件
        echo the files to be removed are: $tar_dir_blobs"*"
        # 如果项目下存在通配符，则需要使用${var}的方式引用
        rm -rf ${tar_dir_blobs}*

        # 然后将snapshots的文件转移到blobs
        tar_dir_snapshots=$root_dir$hub_dir"/snapshots/*/"
        echo the tar_dir_to_snapshots is: $tar_dir_snapshots
        
        mv ${tar_dir_snapshots}* $tar_dir_blobs
        # 转移隐藏文件
        mv ${tar_dir_snapshots}.git* $tar_dir_blobs
        # # 建立blobs下到snapshots/*下的软连接
        # 如下的命令中，使用${tar_dir_snapshots}无法正确解析$root_dir$hub_dir"/snapshots/*/"
        # 因此需要用echo转写一次
        new_dir_snapshots=$(echo $tar_dir_snapshots)
        ls -a $tar_dir_blobs|xargs -I {} ln -s $tar_dir_blobs{} ${new_dir_snapshots}{}
    fi
done

