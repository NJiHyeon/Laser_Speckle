import cv2
import os

#####CLASS1#####
# class1의 동영상에 대한 경로
class1 = []
for c in ['1', '1_2'] :
    for p in ['pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5']:
        PATH = os.path.join('./data/', c, p)
        videofiles = os.listdir(PATH)
        for v in videofiles :
            videofiles_path = os.path.join('./data/', c, p, v)
            print(videofiles_path)
            class1.append(videofiles_path)

# class1의 동영상을 frame으로 추출 
for c1 in class1 :
    vidcap = cv2.VideoCapture(c1)
    success, image = vidcap.read()
    count = 0
    fn = c1.split('/')
    class_name = fn[2]
    pos_name = fn[3]
    video_name = fn[4][:-4]
    foldername = class_name + '_' + pos_name + '_' + video_name
    while success :
        if int(vidcap.get(1))%1==0:
            framename = "f_{number:05}".format(number=count)
            if not os.path.exists(f'./frames/{foldername}'):
                os.makedirs(f'./frames/{foldername}')
            cv2.imwrite(f'./frames/{foldername}/' + framename + '.jpg', image)
            success, image = vidcap.read()
            count += 1
        else :
            success,image = vidcap.read()
            count += 1
            

#=======================================================================================================
#####CLASS2#####        
# class2의 동영상에 대한 경로
class2 = []
for c in ['2', '2_2'] :
    for p in ['pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5']:
        PATH = os.path.join('./data/', c, p)
        videofiles = os.listdir(PATH)
        for v in videofiles :
            videofiles_path = os.path.join('./data', c, p, v)
            class2.append(videofiles_path)

# class2의 동영상을 frame으로 추출
for c2 in class2 :
    vidcap = cv2.VideoCapture(c2)
    success, image = vidcap.read()
    count = 0
    fn = c2.split('/')
    class_name = fn[2]
    pos_name = fn[3]
    video_name = fn[4][:-4]
    foldername = class_name + '_' + pos_name + '_' + video_name
    while success :
        if int(vidcap.get(1))%1==0:
            framename = "f_{number:05}".format(number=count)
            if not os.path.exists(f'./frames/{foldername}'):
                os.makedirs(f'./frames/{foldername}')
            cv2.imwrite(f'./frames/{foldername}/' + framename + '.jpg', image)
            success, image = vidcap.read()
            count += 1
        else :
            success,image = vidcap.read()
            count += 1


#=======================================================================================================
#####CLASS3#####        
# class3의 동영상에 대한 경로
class3 = []
for c in ['3', '3_2'] :
    for p in ['pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5']:
        PATH = os.path.join('./data/', c, p)
        videofiles = os.listdir(PATH)
        for v in videofiles :
            videofiles_path = os.path.join('./data', c, p, v)
            class3.append(videofiles_path)
            
# class3의 동영상을 frame으로 추출
for c3 in class3 :
    vidcap = cv2.VideoCapture(c3)
    success, image = vidcap.read()
    count = 0
    fn = c3.split('/')
    class_name = fn[2]
    pos_name = fn[3]
    video_name = fn[4][:-4]
    foldername = class_name + '_' + pos_name + '_' + video_name
    while success :
        if int(vidcap.get(1))%1==0:
            framename = "f_{number:05}".format(number=count)
            if not os.path.exists(f'./frames/{foldername}'):
                os.makedirs(f'./frames/{foldername}')
            cv2.imwrite(f'./frames/{foldername}/' + framename + '.jpg', image)
            success, image = vidcap.read()
            count += 1
        else :
            success,image = vidcap.read()
            count += 1


#=======================================================================================================
#####CLASS4#####        
# class4의 동영상에 대한 경로
class4 = []
for c in ['4', '4_2'] :
    for p in ['pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5']:
        PATH = os.path.join('./data/', c, p)
        videofiles = os.listdir(PATH)
        for v in videofiles :
            videofiles_path = os.path.join('./data', c, p, v)
            class4.append(videofiles_path)
            
# class4의 동영상을 frame으로 추출
for c4 in class4 :
    vidcap = cv2.VideoCapture(c4)
    success, image = vidcap.read()
    count = 0
    fn = c4.split('/')
    class_name = fn[2]
    pos_name = fn[3]
    video_name = fn[4][:-4]
    foldername = class_name + '_' + pos_name + '_' + video_name
    while success :
        if int(vidcap.get(1))%1==0:
            framename = "f_{number:05}".format(number=count)
            if not os.path.exists(f'./frames/{foldername}'):
                os.makedirs(f'./frames/{foldername}')
            cv2.imwrite(f'./frames/{foldername}/' + framename + '.jpg', image)
            success, image = vidcap.read()
            count += 1
        else :
            success,image = vidcap.read()
            count += 1


#=======================================================================================================
#####CLASS5#####        
# class5의 동영상에 대한 경로
class5 = []
for c in ['5', '5_2'] :
    for p in ['pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5']:
        PATH = os.path.join('./data/', c, p)
        videofiles = os.listdir(PATH)
        for v in videofiles :
            videofiles_path = os.path.join('./data', c, p, v)
            class5.append(videofiles_path)
            
# class5의 동영상을 frame으로 추출
for c5 in class5 :
    vidcap = cv2.VideoCapture(c5)
    success, image = vidcap.read()
    count = 0
    fn = c5.split('/')
    class_name = fn[2]
    pos_name = fn[3]
    video_name = fn[4][:-4]
    foldername = class_name + '_' + pos_name + '_' + video_name
    while success :
        if int(vidcap.get(1))%1==0:
            framename = "f_{number:05}".format(number=count)
            if not os.path.exists(f'./frames/{foldername}'):
                os.makedirs(f'./frames/{foldername}')
            cv2.imwrite(f'./frames/{foldername}/' + framename + '.jpg', image)
            success, image = vidcap.read()
            count += 1
        else :
            success,image = vidcap.read()
            count += 1


#=======================================================================================================
#####CLASS6#####        
# class6의 동영상에 대한 경로
class6 = []
for c in ['6', '6_2'] :
    for p in ['pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5']:
        PATH = os.path.join('./data/', c, p)
        videofiles = os.listdir(PATH)
        for v in videofiles :
            videofiles_path = os.path.join('./data', c, p, v)
            class6.append(videofiles_path)
            
# class6의 동영상을 frame으로 추출
for c6 in class6 :
    vidcap = cv2.VideoCapture(c6)
    success, image = vidcap.read()
    count = 0
    fn = c6.split('/')
    class_name = fn[2]
    pos_name = fn[3]
    video_name = fn[4][:-4]
    foldername = class_name + '_' + pos_name + '_' + video_name
    while success :
        if int(vidcap.get(1))%1==0:
            framename = "f_{number:05}".format(number=count)
            if not os.path.exists(f'./frames/{foldername}'):
                os.makedirs(f'./frames/{foldername}')
            cv2.imwrite(f'./frames/{foldername}/' + framename + '.jpg', image)
            success, image = vidcap.read()
            count += 1
        else :
            success,image = vidcap.read()
            count += 1