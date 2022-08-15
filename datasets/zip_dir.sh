#! /bin/sh

# Choose one path to compress videos
# folder='/mnt/WORKSPACE/E2FGVI-hao/datasets/davis/JPEGImages'
folder="./davis/JPEGImages/480p"
# folder='./datasets/davis/JPEGImages'
# folder='./datasets/youtube-vos/JPEGImages'

if [ -d "$folder" ];then
    for file in $folder/*
    do
        if test -f $file
        then
            echo $file is file
        else
            echo compressing \"$file\"...
            zip -q -r -j $file.zip $file/
            rm -rf $file/
        fi
    done
else
    echo '['$folder']' 'is not exist. Please check the directory.'
fi

echo 'Done!'