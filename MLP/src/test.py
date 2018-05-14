import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
import traceback,psycopg2

# x_false = np.array([[269870.73,271070.74,359581.19,359581.19,0.0,492848.73,493648.75,30569.23,30569.23,0.0,258594.93,259794.93,28154.23,28154.23,0.0,392982.61,393582.61,135161.79,135161.79,0.0]])
# x_true = np.array([[155988.14,156588.15,370501.56,370501.56,0.0,156538.19,156538.19,295401.56,378901.56,90.0,155988.14,156588.15,369301.56,369301.56,0.0,156038.18,156038.18,295401.56,378901.56,90.0]])
# x_undefined = np.array([[136188.22,136788.23,294901.56,294901.56,0.0,523998.71,525198.73,-3030.77,-3030.77,0.0,501907.15,503507.12,390673.35,390673.35,0.0,156002.26,156002.26,118961.79,122261.79,90.0]])
# x_test_ = np.array([[269870.73,271070.74,359581.19,359581.19,0.0,492848.73,493648.75,30569.23,30569.23,0.0,258594.93,259794.93,28154.23,28154.23,0.0,392982.61,393582.61,135161.79,135161.79,0.0]
#         ,[283170.74, 283170.74, 274188.69, 279188.69, 90.0, 272720.74, 273620.74, 369196.19, 369196.19, 0.0, 278270.73, 278870.7, 282788.69, 282788.69, 0.0, 278870.72, 280070.67, 303181.19, 303181.19, 0.0]])
# x_unknown=preprocessing.StandardScaler().fit_transform(x_unknown)
sn_ = [3233119, 3233120, 3233131, 3233134, 3234783, 3234784, 3234785, 3234786, 3234787, 3234800]
sn_group = [155988.14,156588.15,370501.56,370501.56,0.0,156538.19,156538.19,295401.56,378901.56,90.0,155988.14,156588.15,369301.56,369301.56,0.0,156038.18,156038.18,295401.56,378901.56,90.0]

def cal_line(x1,x2,y1,y2):
    a = y1-y2
    b = x2-x1
    c = x1*y2-x2*y1
    return a,b,c

sn_array = np.array(sn_group).reshape((-1,5))
sn_sort = sn_array[np.lexsort(sn_array.T)]

cross = []

for i in (0,1):
    a0,b0,c0 = cal_line(sn_sort[i][0], sn_sort[i][1],
                        sn_sort[i][2], sn_sort[i][3])

    for i in (2,3):
        a1,b1,c1 = cal_line(sn_sort[i][0], sn_sort[i][1],
                        sn_sort[i][2], sn_sort[i][3])
        D = a0*b1-a1*b0
        v_x = (b0*c1-b1*c0)/D
        v_y = (a1*c0 - a0*c1)/D
        cross.append(v_x)
        cross.append(v_y)

x = cross[::2]
y = cross[1::2]

cross = np.array(cross).reshape(-1,2)

v1 = {'x':cross[0][0], 'y':cross[0][1]}
v2 = {'x':cross[1][0], 'y':cross[2][1]}
v3 = {'x':cross[2][0], 'y':cross[3][1]}
v4 = {'x':cross[3][0], 'y':cross[3][1]}


# cross1 = cross[:,::-1].T
# cross2 = np.lexsort(cross1)
# cross_sorted= cross[cross2]
# print(cross_sorted)

# ax = plt.subplot(aspect='equal')
#
# ax.scatter(x,y,linewidth=0.1)
# ax.axis('off')
# plt.show()





