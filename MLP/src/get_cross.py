import psycopg2,time,math
import traceback
import matplotlib.pyplot as plt

def con_db(db_domain, db_name, db_user, db_ps, db_port):
    connection = psycopg2.connect(host=db_domain,
                                  dbname=db_name,
                                  user=db_user,
                                  password=db_ps,
                                  port=db_port)
    return connection

def close_db(conn, cursor):
    conn.commit()
    cursor.close()
    conn.close()

#计算直线系数
def cal_line(x1,x2,y1,y2):
    a = y1-y2
    b = x2-x1
    c = x1*y2-x2*y1
    return a,b,c

def get_cross():
    print("calculating")
    try:
        conn = con_db("192.168.217.129", "exampledb", "dever", "dever", "5432")
        # conn = con_db("192.168.163.129", "lm", "dever", "dever", "5432")
        curs = conn.cursor()
        sql = "select * from test_table;"

        curs.execute(sql)

        rows = curs.fetchall()

        vertexes = []
        fig = plt.figure()
        ax = plt.subplot(aspect='equal')
        ax.axis('off')


        for record in rows:
            cross = []
            a0, b0, c0 = cal_line(record[0]['x1'], record[0]['x2'],
                                  record[0]['y1'], record[0]['y2'])

            for i in (2, 3):
                a1, b1, c1 = cal_line(record[i]['x1'], record[i]['x2'],
                                  record[i]['y1'], record[i]['y2'])
                D = a0 * b1 - a1 * b0
                v_x = (b0 * c1 - b1 * c0) / D
                v_y = (a1 * c0 - a0 * c1) / D
                cross.append(round(v_x,2))
                cross.append(round(v_y,2))

            a0, b0, c0 = cal_line(record[1]['x1'], record[1]['x2'],
                                  record[1]['y1'], record[1]['y2'])

            for i in (3, 2):
                a1, b1, c1 = cal_line(record[i]['x1'], record[i]['x2'],
                                  record[i]['y1'], record[i]['y2'])
                D = a0 * b1 - a1 * b0
                v_x = (b0 * c1 - b1 * c0) / D
                v_y = (a1 * c0 - a0 * c1) / D
                cross.append(round(v_x,2))
                cross.append(round(v_y,2))

            v_x = cross[::2]
            v_y = cross[1::2]

            v_x.append(cross[0])
            v_y.append(cross[1])
            ax.plot(v_x,v_y,linewidth=0.1)
            vertexes.append(cross)

        # print(vertexes[0])


        fig.savefig("./frame.png",dpi = 2000)


        plt.show()

    except Exception as e:
        print(traceback.print_exc())
    finally:
        close_db(conn, curs)

get_cross()

