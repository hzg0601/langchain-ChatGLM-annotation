# 软连接所在目录
target_dir=/home/huangzhiguo/.cache/huggingface/hub/models--moka-ai--m3e-base/snapshot/9fb6b56338045322b8496054dc8768a770dc9583/
# 文件实际目录
source_dir=/home/huangzhiguo/.cache/huggingface/hub/models--moka-ai--m3e-base/blobs/
# 获取文件实际目录的所有文件
files=$(ls -a $source_dir)

for file in $files
do
    if [ $file != $(basename "$0") ];then
        source_file=$source_dir$file
        target_link=$target_dir$file
        echo $source_file
        echo $target_link
        if [ -e source_file ];then
            ln -s $source_file $target_link
        else
            echo "source_file $source_file doesn't exist."
        fi
    fi
done