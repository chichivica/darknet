#!/bin/bash

for file in /DATA/autopass/door/2019-01-21/*;
do
#    echo $file
    b_file=$(basename $file)
    path=${file/$b_file/}
    echo $b_file
    echo $path
    avi_file=${file/.mp4/.avi}

    mkdir $path/output || true

    ./darknet detector demo myconfig/autopass.data myconfig/yolov3-autopass.cfg backup/yolov3-autopass_18000.weights $file -thresh 0.25 -prefix ${file/.mp4/.avi} -i 3
    ffmpeg -i $avi_file $path/output/$b_file

    txt_file=$path/output/$b_file.txt;

    mv $avi_file.txt $txt_file;
    rm $avi_file

    python3 python/convert_for_markup_tool.py $txt_file

done