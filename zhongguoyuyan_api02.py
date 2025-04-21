# 批量转换Bilibili视频链接为指定格式的脚本

# 原始链接列表
original_urls = [
    "https://www.bilibili.com/video/BV1Si4y1B7ZX",
"https://www.bilibili.com/video/BV1va4y197Xt",
"https://www.bilibili.com/video/BV1rH4y1X7Jk",
"https://www.bilibili.com/video/BV1iE4m1R7iq",
"https://www.bilibili.com/video/BV1a7WVeGEGg",
"https://www.bilibili.com/video/BV1WF411k7ss",
"https://www.bilibili.com/video/BV1bb421n76k",
"https://www.bilibili.com/video/BV1q94y147dc",
"https://www.bilibili.com/video/BV16W421974y",
"https://www.bilibili.com/video/BV1Nw411q7HA",
"https://www.bilibili.com/video/BV1S94y1k7By",
"https://www.bilibili.com/video/BV1hk4y1U7N5",
"https://www.bilibili.com/video/BV1Eh4y1q787",
"https://www.bilibili.com/video/BV1r1421i7uC",
"https://www.bilibili.com/video/BV1C9WGeZEjo",
"https://www.bilibili.com/video/BV1fH4y1o7ac",
"https://www.bilibili.com/video/BV1w8411q7FY",
"https://www.bilibili.com/video/BV1Au4y1r7Ht",
"https://www.bilibili.com/video/BV1CN411i7Wn",
"https://www.bilibili.com/video/BV12r4y1R7HJ",
"https://www.bilibili.com/video/BV1c14y1y7Wf",
"https://www.bilibili.com/video/BV1CENhefEMm",
"https://www.bilibili.com/video/BV1Lv411F7Sj",
"https://www.bilibili.com/video/BV1s14y1y7EW",
"https://www.bilibili.com/video/BV1cm411S78A",
"https://www.bilibili.com/video/BV1dN411v7eG",
"https://www.bilibili.com/video/BV1Pm421372q",
"https://www.bilibili.com/video/BV1b1421b7eM",
"https://www.bilibili.com/video/BV191421X785",
"https://www.bilibili.com/video/BV1dz4y1M7Dt",
"https://www.bilibili.com/video/BV1CENhefEMm/",
"https://www.bilibili.com/video/BV1C9WGeZEjo/",
"https://www.bilibili.com/video/BV1a7WVeGEGg/",
"https://www.bilibili.com/video/BV1Pm421372q/",
"https://www.bilibili.com/video/BV1iE4m1R7iq/",
"https://www.bilibili.com/video/BV16W421974y/",
"https://www.bilibili.com/video/BV1bb421n76k/",
"https://www.bilibili.com/video/BV1b1421b7eM/",
"https://www.bilibili.com/video/BV1r1421i7uC/",
"https://www.bilibili.com/video/BV191421X785/",
"https://www.bilibili.com/video/BV1cm411S78A/",
"https://www.bilibili.com/video/BV1hk4y1U7N5/",
"https://www.bilibili.com/video/BV1Si4y1B7ZX/",
"https://www.bilibili.com/video/BV1va4y197Xt/",
"https://www.bilibili.com/video/BV1Lv411F7Sj/",
"https://www.bilibili.com/video/BV1Eh4y1q787/",
"https://www.bilibili.com/video/BV1fH4y1o7ac/",
"https://www.bilibili.com/video/BV1Au4y1r7Ht/",
"https://www.bilibili.com/video/BV1Nw411q7HA/",
"https://www.bilibili.com/video/BV1w8411q7FY/",
"https://www.bilibili.com/video/BV1dN411v7eG/",
"https://www.bilibili.com/video/BV1rH4y1X7Jk/",
"https://www.bilibili.com/video/BV1WF411k7ss/",
"https://www.bilibili.com/video/BV1s14y1y7EW/",
"https://www.bilibili.com/video/BV1q94y147dc/",
"https://www.bilibili.com/video/BV1c14y1y7Wf/",
"https://www.bilibili.com/video/BV1CN411i7Wn/",
"https://www.bilibili.com/video/BV12r4y1R7HJ/",
"https://www.bilibili.com/video/BV1dz4y1M7Dt/",
"https://www.bilibili.com/video/BV1S94y1k7By/",
"https://www.bilibili.com/video/BV1cP411k74x",
"https://www.bilibili.com/video/BV1Yu4y1U7Ga",
"https://www.bilibili.com/video/BV1hX4y177XV",
"https://www.bilibili.com/video/BV1Qu411G7aT",
"https://www.bilibili.com/video/BV1y84y1Z7K7",
"https://www.bilibili.com/video/BV1AV4y187ER",
"https://www.bilibili.com/video/BV1Lc411w7SA",
"https://www.bilibili.com/video/BV1RN41187Cx"
]

# 目标替换链接
replacement_url = "https://www.bilibili.com/video/BV13g411j7Ys/?spm_id_from=333.337.search-card.all.click"

# 转换结果
converted_links = {
    replacement_url: "橘柚Nuyoah"
}

# 打印转换结果
for url, name in converted_links.items():
    print(f'"{url}": "{name}"')

# 如果需要，可以将结果写入文件
with open('converted_links.txt', 'w', encoding='utf-8') as f:
    for url, name in converted_links.items():
        f.write(f'"{url}": "{name}"\n')

print("\n转换完成。结果已保存到 converted_links.txt 文件中。")