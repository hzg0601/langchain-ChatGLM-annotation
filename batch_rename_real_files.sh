#!/bin/bash
files=$(ls -a ./)
# 本脚本需置于软链接所在目录
for file in $files: 
do  
    # basename：命令本身 "$0"：代表当前脚本的文件名，$0是一个特殊变量，它包含当前脚本的文件名。
    if test "$file" != "$(basename "$0")";then
        echo $file
        # readlink -f 获取文件指向的真实地址
        pointed_name=$(readlink -f $file)
        # 获取文件指向真实地址所在目录

        pointed_dir=$(dirname $pointed_name)
        # # 更改文件名 vocab.txt后会跟一个:，不清楚是为什么
        # 使用${parameter/pattern/string}语法删除mystr字符串中的所有字符l。
        # 其中，//表示全局替换，/后面为空字符串，表示将匹配到的字符删除。
        echo $pointed_dir   
        file=${file//:/}
        # 真实文件的真实文件名
        readable_name="$pointed_dir/$file"
        echo "the file name to be renamed: $readable_name"
        echo "****************************"
        echo "the file name with random str:$pointed_name"
        # 如果被指向的文件存在，才进行重命名
        if [ -e $pointed_name ];then
            # 重命名
            mv $pointed_name $readable_name
            
        fi
    fi
done
echo "rename succeed"
# echo $(basename "$0")