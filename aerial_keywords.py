#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
航拍关键词集合
包含自然风光、城市景观、特殊场景等多个类别
"""

# 自然风光航拍关键词
NATURE_AERIAL_KEYWORDS = [
    "山脉航拍", "森林航拍", "海岸线航拍", "河流航拍", "瀑布航拍",
    "草原航拍", "沙漠航拍", "湖泊航拍", "冰川航拍", "火山航拍",
    "峡谷航拍", "海滩航拍", "岛屿航拍", "雪山航拍", "热带雨林航拍",
    "珊瑚礁航拍", "沼泽航拍", "绿洲航拍", "高原航拍", "丘陵航拍",
    "溪流航拍", "海湾航拍", "荒野航拍", "峭壁航拍", "山谷航拍"
]

# 城市景观航拍关键词
CITY_AERIAL_KEYWORDS = [
    "城市天际线航拍", "现代建筑群航拍", "古建筑群航拍", "城市公园航拍",
    "商业区航拍", "住宅区航拍", "工业区航拍", "港口航拍", "机场航拍",
    "火车站航拍", "体育场馆航拍", "大学校园航拍", "城市广场航拍",
    "城市交通航拍", "桥梁航拍", "地标建筑航拍", "夜景航拍", "城市绿地航拍",
    "滨水区航拍", "城市规划区航拍", "商圈航拍", "文化中心航拍",
    "城市基建航拍", "城市更新区航拍", "特色街区航拍"
]

# 农业景观航拍关键词
AGRICULTURE_AERIAL_KEYWORDS = [
    "农田航拍", "梯田航拍", "果园航拍", "茶园航拍", "稻田航拍",
    "花田航拍", "葡萄园航拍", "农场航拍", "牧场航拍", "温室大棚航拍",
    "灌溉系统航拍", "农业基地航拍", "养殖场航拍", "农业科技园航拍",
    "观光农业航拍", "生态农业航拍", "有机农场航拍", "现代农业航拍",
    "特色农业航拍", "农业产业园航拍"
]

# 特殊场景航拍关键词
SPECIAL_AERIAL_KEYWORDS = [
    "日出航拍", "日落航拍", "雾霾航拍", "云海航拍", "极光航拍",
    "星空航拍", "雪景航拍", "雨景航拍", "暴风雨航拍", "彩虹航拍",
    "闪电航拍", "沙尘暴航拍", "台风航拍", "洪水航拍", "火灾航拍",
    "地震灾区航拍", "火山喷发航拍", "海啸航拍", "冰雪消融航拍",
    "环境污染航拍"
]

# 文化遗产航拍关键词
HERITAGE_AERIAL_KEYWORDS = [
    "古城航拍", "古镇航拍", "寺庙航拍", "宫殿航拍", "长城航拍",
    "古墓航拍", "考古遗址航拍", "文物保护区航拍", "历史街区航拍",
    "古建筑群航拍", "文化遗产航拍", "古村落航拍", "古塔航拍",
    "古城墙航拍", "古园林航拍", "古桥航拍", "古驿站航拍",
    "古战场航拍", "古港口航拍", "古驿道航拍"
]

# 交通设施航拍关键词
TRANSPORTATION_AERIAL_KEYWORDS = [
    "高速公路航拍", "铁路航拍", "立交桥航拍", "隧道航拍", "码头航拍",
    "机场跑道航拍", "地铁站航拍", "公交枢纽航拍", "停车场航拍",
    "物流中心航拍", "高铁站航拍", "公路网航拍", "航运航拍",
    "交通枢纽航拍", "轨道交通航拍", "客运站航拍", "货运站航拍",
    "交通管制航拍", "收费站航拍", "服务区航拍"
]

# 工业设施航拍关键词
INDUSTRIAL_AERIAL_KEYWORDS = [
    "工厂航拍", "发电站航拍", "矿场航拍", "钢铁厂航拍", "化工厂航拍",
    "工业园区航拍", "仓储基地航拍", "制造业基地航拍", "能源基地航拍",
    "工业港口航拍", "工业废弃地航拍", "工业遗址航拍", "工业园航拍",
    "高新技术园区航拍", "生态工业园航拍", "循环经济园区航拍",
    "产业集群航拍", "工业新城航拍", "科技园区航拍", "创新产业园航拍"
]

# 体育设施航拍关键词
SPORTS_AERIAL_KEYWORDS = [
    "体育场航拍", "足球场航拍", "篮球场航拍", "网球场航拍", "游泳馆航拍",
    "体育中心航拍", "运动场航拍", "健身公园航拍", "滑雪场航拍",
    "高尔夫球场航拍", "马术场航拍", "赛车场航拍", "自行车道航拍",
    "极限运动场航拍", "水上运动中心航拍", "体育训练基地航拍",
    "运动公园航拍", "体育小镇航拍", "全民健身中心航拍", "体育产业园航拍"
]

# 旅游景点航拍关键词
TOURISM_AERIAL_KEYWORDS = [
    "景区航拍", "主题公园航拍", "度假村航拍", "游乐园航拍", "观光区航拍",
    "旅游景点航拍", "风景区航拍", "旅游度假区航拍", "生态旅游区航拍",
    "文化旅游区航拍", "休闲度假区航拍", "旅游小镇航拍", "旅游街区航拍",
    "特色景区航拍", "旅游景观航拍", "旅游集散地航拍", "旅游目的地航拍",
    "旅游产业园航拍", "旅游新区航拍", "旅游综合体航拍"
]

# 所有航拍关键词合集
ALL_AERIAL_KEYWORDS = (
    NATURE_AERIAL_KEYWORDS +
    CITY_AERIAL_KEYWORDS +
    AGRICULTURE_AERIAL_KEYWORDS +
    SPECIAL_AERIAL_KEYWORDS +
    HERITAGE_AERIAL_KEYWORDS +
    TRANSPORTATION_AERIAL_KEYWORDS +
    INDUSTRIAL_AERIAL_KEYWORDS +
    SPORTS_AERIAL_KEYWORDS +
    TOURISM_AERIAL_KEYWORDS
)

# 获取所有关键词数量
TOTAL_KEYWORDS = len(ALL_AERIAL_KEYWORDS)

if __name__ == "__main__":
    print(f"总共收集了 {TOTAL_KEYWORDS} 个航拍关键词")
    print("\n各类别关键词数量：")
    print(f"自然风光：{len(NATURE_AERIAL_KEYWORDS)}")
    print(f"城市景观：{len(CITY_AERIAL_KEYWORDS)}")
    print(f"农业景观：{len(AGRICULTURE_AERIAL_KEYWORDS)}")
    print(f"特殊场景：{len(SPECIAL_AERIAL_KEYWORDS)}")
    print(f"文化遗产：{len(HERITAGE_AERIAL_KEYWORDS)}")
    print(f"交通设施：{len(TRANSPORTATION_AERIAL_KEYWORDS)}")
    print(f"工业设施：{len(INDUSTRIAL_AERIAL_KEYWORDS)}")
    print(f"体育设施：{len(SPORTS_AERIAL_KEYWORDS)}")
    print(f"旅游景点：{len(TOURISM_AERIAL_KEYWORDS)}")

# Use all keywords
for keyword in ALL_AERIAL_KEYWORDS:
    print(keyword)

# Use specific category keywords
for keyword in NATURE_AERIAL_KEYWORDS:
    print(keyword) 