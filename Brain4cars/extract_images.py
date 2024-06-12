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
