import cv2
def draw_up_down_counter(img, up_counter, down_counter, frame_feature, names):
    '''cv2.rectangle(img, (0, 0), (520, 220), (255, 255, 255), thickness=-1)
    cv2.putText(img, 'veh_type', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    text_size = 2*cv2.getTextSize('veh_type', cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, thickness=-1)
    cv2.putText(img, 'up', (int(text_size[0][0]) + 60, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)

    cv2.putText(img, 'down', (int(text_size[0][0]) + 160, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    for i, name in enumerate(names):
        cv2.putText(img, '%s' %name, (10, (i+2)*40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,0),4)
        cv2.putText(img, '%s' %str(up_counter[i]), ((int(text_size[0][0]) + 60), (i+2)*40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
        cv2.putText(img, '%s' %str(down_counter[i]), ((int(text_size[0][0]) + 160), (i+2) * 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    cv2.putText(img,'Total',(10,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    cv2.putText(img, '%s' %str(sum(up_counter)),(int(text_size[0][0]) + 60,200),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,0),4)
    cv2.putText(img, '%s' % str(sum(down_counter)), (int(text_size[0][0]) + 160,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 0, 0), 4)
'''
    cv2.rectangle(img, (0, 0), (260, 110), (255, 255, 255), thickness=-1)
    cv2.putText(img, 'veh_type', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    text_size = cv2.getTextSize('veh_type', cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, thickness=-1)
    cv2.putText(img, 'up', (int(text_size[0][0]) + 30, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    #up_counter=[0,0,0]
    #down_counter=[down_counter[0],down_counter[1],3]
    #up_counter=[up_counter[0],up_counter[1],1]
    cv2.putText(img, 'down', (int(text_size[0][0]) + 80, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    for i, name in enumerate(names):
        cv2.putText(img, '%s' %name, (10, (i+2)*20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),2)
        cv2.putText(img, '%s' %str(up_counter[i]), ((int(text_size[0][0]) + 30), (i+2)*20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(img, '%s' %str(down_counter[i]), ((int(text_size[0][0]) + 80), (i+2) * 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img,'Total',(10,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img, '%s' %str(sum(up_counter)),(int(text_size[0][0]) + 30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),2)
    cv2.putText(img, '%s' % str(sum(down_counter)), (int(text_size[0][0]) + 80,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 0), 2)

    '''for i, name in enumerate(names):
        cv2.putText(img, 'Up %s :' %name + str(up_counter[i]), (10, (i+1)*25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),1 )
        cv2.putText(img, 'Down %s :' %name + str(down_counter[i]), (10, (i+4)*25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)'''
