'''
使用在datasets/annotation中的脚本extract_frames.py来提取图像:将这个文件复制到“face_camera”目录下，然后运行这个脚本。
将.avi格式的视频文件转换为一系列帧图像，并保存在与视频文件相同的目录中。

具体来说，它使用os.walk()函数遍历指定目录（'../../datasets/annotation/'）下的所有文件，
找到以.avi结尾的视频文件，生成一个包含视频文件路径和所在目录路径的列表。
然后，对于每个视频文件，它将输出图像的路径设置为在同一目录中，名称为image-%4d.png，其中%4d表示帧编号。
最后，它使用ffmpeg命令将视频文件转换为图像序列，输出路径由前面设置的output变量指定。

'''
import os

videos = []
paths = []
for path, dirs, files in os.walk('E://86182//Documents//CQUT//doing//task//Rong//Driver-Intention-Prediction-master//Brain4Cars//img//zenodo//'):
    for f in files:
        if f.endswith('.mov'):
            videos.append(os.path.join(path, f))
            paths.append(path)

for i in range (0,len(videos)):
    filename = videos[i]
    path = paths[i]
    output = os.path.join(path,'image-%4d.png')
    print(output)
    os.system("ffmpeg -i {0} -f image2 -vf fps=25 {1}".format(filename,output))
