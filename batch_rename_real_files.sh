files=$(ls -a ./)
for file in $files: 
do  
    # basename：命令本身 "$0"：代表当前脚本的文件名，$0是一个特殊变量，它包含当前脚本的文件名。
    if test "$file" != "$(basename "$0")";then
        echo $file
        # readlink -f 获取文件指向的路径
        pointed_name=$(readlink -f $file)
        pointed_dir=$(readlink -f $file|xargs dirname)
        # vocab.txt后会跟一个:，不清楚是为什么
        # 使用${parameter/pattern/string}语法删除mystr字符串中的所有字符l。其中，//表示全局替换，/后面为空字符串，表示将匹配到的字符删除。
        file=${file//:/}
        readable_name="$pointed_dir/$file"
        echo $readable_name
        echo "****************************"
        echo $pointed_name
        # 如果被指向的文件存在，才进行重命名
        if [ -e $pointed_name ];then
            mv $pointed_name $readable_name
            
        fi
    fi
done

# echo $(basename "$0")