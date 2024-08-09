#下载pyecharts以及相关的地图包
# echarts-china-provinces-pypkg 中国省级地图
# echarts-china-cities-pypkg 中国市级地图
# echarts-china-misc-pypkg 中国区域地图 如：华南、华北

import pandas
from pyecharts.charts import Geo
from pyecharts import options as opts

# province_distribution={'北京':1876,'上海':1095,'湖北':672,'广州':1876,'云南':4,'黑龙江':1876,
#                        '吉林':1876,'广东':2023,'海南':3,'福建':1876,'浙江':1876,'江苏':1876,'河北':1876,
#                        '天津':1876,'四川':723,'重庆':1876,'贵州':1876,'湖南':1876,'山西':1876,'安徽':1876}

#1.读取文件，并且将Dataframe数据转换为字典数据
def get_excel(file_path):
    #pandas读取Excel表格数据
    data_frame=pandas.read_excel(file_path)
    #将Dataframe转为列表-字典的数据格式
    #返回的是列表包含字典的数据
    dict_list=data_frame.to_list('record')
    """每个字典：
        {'岗位名称':'项目经理',
        '岗位链接':'https://www.nowcoder.com/fulltime/42343?jobIds=50777&ncsr=',
        '公司名称':'平安科技（深圳）有限公司',
        '地点':'上海,深圳',
        '工资':'¥薪资面议',
        '时间':'1天前',
        '岗位职责':'...',
        '岗位要求':'...',
        'label':'项目经理',}"""

#2.画地图
def draw_geo(dict_list):
    geography=Geo()

    #设置要显示的地图：长沙
    geography.add_schema(maptype=city)
    #定义数据对：传入地图，并且显示在地图上的数据（公司名称和岗位名称）
    data_pair=[]

    for company_dict in dict_list:
        #地图上要显示的公司名称和岗位名称
        name=company_dict.get('岗位名称')+'('+company_dict.get('公司名称')+')'
        #每个职位的位置
        geography.add_coordinate(name,company_dict.get('地点'))
        data_pair.append((name,company_dict.get('工资')))

    #将数据添加到地图上
    #1.岗位地点：按钮一按就消失或出来
    #2.data_pair:要在地图上显示的数据
    #3.type_:Geo图类型，有scatter（散点图），effectScatter（涟漪散点图），
    #                    heatmap（热力图），lines（流向图）
    #GeoType GeoType.EFFECT_SCATTER, GeoType.HEATMAP, GeoType.LINES

    geography.add('职位位置',data_pair,type_=GeoType.EFFECT_SCATTER,symbol_size=5)

    geography.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(is_piecewise=True),
        title_opts=opts.TitleOpts(title='{}')
    )
    return geography

if __name__=='__main__':
    file_path=r'工作地点.xlsx'
    dict_list=get_excel(file_path)
    geography.render('Chinawork.html')



