import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import psycopg2,time,math
import traceback,itertools
import src.main as model

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

##横竖线组合生成器
def group_gene(hen_,shu_):
    for hen in hen_:
        hen_min_x = min(hen[0], hen[1], hen[5], hen[6])
        hen_max_x = max(hen[0], hen[1], hen[5], hen[6])
        hen_min_y = min(hen[2], hen[3], hen[7], hen[8])
        hen_max_y = max(hen[2], hen[3], hen[7], hen[8])

        for shu in shu_:
            shu_max_x = max(shu[0], shu[1], shu[5], shu[6])
            shu_min_x = min(shu[0], shu[1], shu[5], shu[6])
            shu_max_y = max(shu[2], shu[3], shu[7], shu[8])
            shu_min_y = min(shu[2], shu[3], shu[7], shu[8])

            if shu_max_x > hen_min_x and shu_min_x < hen_max_x and shu_max_y > hen_min_y and shu_min_y < hen_max_y:
                yield shu+hen

def group_gene_dict(hen_,shu_,hen_dic,shu_dic):
    for hen in hen_:
        hen_min_x = min(hen_dic[hen[0]][0], hen_dic[hen[0]][1], hen_dic[hen[1]][0], hen_dic[hen[1]][1])
        hen_max_x = max(hen_dic[hen[0]][0], hen_dic[hen[0]][1], hen_dic[hen[1]][0], hen_dic[hen[1]][1])
        hen_min_y = min(hen_dic[hen[0]][2], hen_dic[hen[0]][3], hen_dic[hen[1]][2], hen_dic[hen[1]][3])
        hen_max_y = max(hen_dic[hen[0]][2], hen_dic[hen[0]][3], hen_dic[hen[1]][2], hen_dic[hen[1]][3])

        for shu in shu_:
            shu_max_x = max(shu_dic[shu[0]][0], shu_dic[shu[0]][1], shu_dic[shu[1]][0], shu_dic[shu[1]][1])
            shu_min_x = min(shu_dic[shu[0]][0], shu_dic[shu[0]][1], shu_dic[shu[1]][0], shu_dic[shu[1]][1])
            shu_max_y = max(shu_dic[shu[0]][2], shu_dic[shu[0]][3], shu_dic[shu[1]][2], shu_dic[shu[1]][3])
            shu_min_y = min(shu_dic[shu[0]][2], shu_dic[shu[0]][3], shu_dic[shu[1]][2], shu_dic[shu[1]][3])

            if shu_max_x > hen_min_x and shu_min_x < hen_max_x and shu_max_y > hen_min_y and shu_min_y < hen_max_y:
                yield shu+hen

def start():
    try:
        conn = con_db("192.168.217.129", "exampledb", "dever", "dever", "5432")
        # conn = con_db("192.168.163.129", "lm", "dever", "dever", "5432")
        curs = conn.cursor()

        curs.execute("select sn,x1,x2,y1,y2,k from (select * from tb_lm_henshu where facade_id = '1') as f where k < -83 or k > 83;")

        #all sn num
        rows = curs.fetchall()

        shu_dict = {}
        for sn, x1, x2, y1, y2, k in tuple(rows):
            key = sn
            shu_dict.setdefault(key,[]).extend((float(x1),float(x2),float(y1),float(y2),float(k)))


        ite = itertools.combinations(shu_dict,2)

        ####generator
        shu_combination = [
            shu_dict[i[0]] + shu_dict[i[1]]
            for i in ite #限定竖线之间的宽度在100-2000以内
           if abs(shu_dict[i[0]][0] - shu_dict[i[1]][0]) > 100 and abs(shu_dict[i[0]][0] - shu_dict[i[1]][0]) < 2000 \
              and abs(shu_dict[i[0]][1] - shu_dict[i[1]][1]) > 100 and abs(shu_dict[i[0]][1] - shu_dict[i[1]][1]) < 2000 \
              # 竖线1的最大值比竖线2的最小值小或者竖线1的最小值比竖线2的最大值大
              and max(shu_dict[i[0]][2], shu_dict[i[0]][3]) > min(shu_dict[i[1]][3], shu_dict[i[1]][2]) \
              and min(shu_dict[i[0]][2], shu_dict[i[0]][3]) < max(shu_dict[i[1]][3], shu_dict[i[1]][2])]

        # print(shu_combination[0:2])
        #横线部分


        curs.execute("select sn,x1,x2,y1,y2,k from (select * from tb_lm_henshu where facade_id = '1') as f where k < 7 and k > -7;")

        # all sn num
        rows_1 = curs.fetchall()

        hen_dict = {}
        for sn, x1, x2, y1, y2, k in tuple(rows_1):
            key = sn
            hen_dict.setdefault(key, []).extend((float(x1), float(x2), float(y1), float(y2), float(k)))

        ite_hen = itertools.combinations(hen_dict, 2)
        ##genrator
        hen_combination = [ hen_dict[i[0]]+hen_dict[i[1]]
                           for i in ite_hen # 限定横线之间的高度在100-9000以内
               if abs(hen_dict[i[0]][2] - hen_dict[i[1]][2]) > 100 and abs(hen_dict[i[0]][2] - hen_dict[i[1]][2]) < 9000 \
               and abs(hen_dict[i[0]][3] - hen_dict[i[1]][3]) > 100 and abs(hen_dict[i[0]][3] - hen_dict[i[1]][3]) < 9000 \
               # 横线1的最大值比横线2的最小值小或者横线1的最小值比横线2的最大值大
               and max(hen_dict[i[0]][0], hen_dict[i[0]][1]) > min(hen_dict[i[1]][0], hen_dict[i[1]][1]) \
               and min(hen_dict[i[0]][0], hen_dict[i[0]][1]) < max(hen_dict[i[1]][0], hen_dict[i[1]][1])]
        print("横线组合 %d 生成完成" % len(hen_combination))

        # print(hen_combination[0])


        it = iter(itertools.product(shu_combination,hen_combination))

        ##不能形成边框的条件： 1.竖线的最大横坐标小于横线的最小横坐标
        ##                   2.竖线的最小横坐标大于横线的最大横坐标
        ##                   3.竖线的最大纵坐标小于横线的最小纵坐标
        ##                   4.竖线的最小纵坐标大于横线的最大纵坐标
        # group = []
        group = group_gene_dict(hen_combination,shu_combination,hen_dict,shu_dict)

        '''
        group = group_gene(hen_combination,shu_combination)
        frame_need_test= [g for g in group]

        print("初步筛选线框完成，即将投入多层感知机识别")

        length = len(frame_need_test)

        # tmp_test = frame_need_test[:(math.ceil(length/25))]
        # # print(tmp_test[:10])
        # test_label = model.model_start(tmp_test)
        # # print(len(list(test_label)))
        # print(list(test_label).count(1))

        # tmp_test1 = frame_need_test[(math.ceil(length/25)*3):(math.ceil(length/25)*4)]
        # test_label1 = model.model_start(tmp_test1)
        # print(len(test_label1))
        # print(list(test_label1).count(1))
        # sn_combination = []
        # for row in rows:
        #   sn_combination.append(row[0])

        # ite = itertools.combinations(sn_combination[1000:1005],4) #combination iteration
        # sql = "select x1,x2,y1,y2,k from tb_lm_henshu where sn = %s"
        # insert_sql = "insert into "
        # count = 0
        # # sn_group1 = [155988.14, 156588.15, 370501.56, 370501.56, 0.0, 156538.19, 156538.19, 295401.56, 378901.56, 90.0,
        # #              155988.14, 156588.15, 369301.56, 369301.56, 0.0, 156038.18, 156038.18, 295401.56, 378901.56, 90.0]
        # # model.model_start(sn_group1)
        # for item in ite:
        #     group = []
        #     item = list(item)
        #     for i in item:
        #         curs.execute(sql % i)
        #         res = curs.fetchall()
        #         for k in res[0]:
        #             group.append(float(k))
        #     print("iteration: %d" % count)
        #     print(group)
        #     model.model_start(group)
        #     count+=1
    '''
    except Exception as e:
        print(traceback.print_exc())
    finally:
        close_db(conn, curs)

start_time = time.clock()
start()
print("Time cost: %s" % round(time.clock()-start_time,2))
