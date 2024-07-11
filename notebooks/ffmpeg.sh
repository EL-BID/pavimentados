#/bin/sh
video1=base.mp4 &&
video2=$1.mp4 &&
output=base_$1.mp4 &&
text1="yolov3" &&
text2="$1" &&
ffmpeg -y -i $video1 -i $video2 -filter_complex hstack tmp.mp4 &&
ffmpeg -y -i tmp.mp4 -vf "drawtext=fontfile=/path/to/font.ttf:text='$text1':fontcolor=white:fontsize=14:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=h-th-10,drawtext=fontfile=/path/to/font.ttf:text='$text2':fontcolor=white:fontsize=14:box=1:boxcolor=black@0.5:boxborderw=5:x=w-tw-10:y=h-th-10"  -codec:a copy $output
rm tmp.mp4

#ffplay -vf "
#drawtext=fontfile=/path/to/font.ttf:text='current version':
#fontcolor=white:fontsize=14:box=1:boxcolor=black@0.5:boxborderw=5:
#x=10:y=h-th-10,

#drawtext=fontfile=/path/to/font.ttf:text='conf0.1_iou0.2':
#fontcolor=white:fontsize=14:box=1:boxcolor=black@0.5:boxborderw=5:
#x=w-tw-10:y=h-th-10
#" 
#-codec:a copy bench_conf0.1_iou0.2.mp4