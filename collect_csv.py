import requests
from bs4 import BeautifulSoup
import os
path = './ssq.txt'

if os.path.exists(path):
    os.remove(path)
else:
    print("not file！")

def save_to_file(content):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')



basic_url = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list_1.html'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'
}
response = requests.get(basic_url, headers=headers, timeout=10)
response.encoding = 'utf-8'
htm = response.text

# 解析内容
soup = BeautifulSoup(htm, 'html.parser')

# 获取页数信息
page = int(soup.find('p', attrs={"class": "pg"}).find_all('strong')[0].text)


#接下来，我们就可以根据规律组装好我们的URL：

url_part = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list'

for i in range(1, page+1):
    url = url_part + '_' + str(i) + '.html'



    res = requests.get(url, headers=headers, timeout=10)
    res.encoding = 'utf-8'
    context = res.text
    soups = BeautifulSoup(context, 'html.parser')

    if soups.table is None:
        continue
    elif soups.table:
        table_rows = soups.table.find_all('tr')
        for row_num in range(2, len(table_rows)-1):
            row_tds = table_rows[row_num].find_all('td')
            ems = row_tds[2].find_all('em')
            result = row_tds[0].string +', '+ row_tds[1].string +', '+ems[0].string+' '+ems[1].string+' '+ems[2].string+' '+ems[3].string+' '+ems[4].string+' '+ems[5].string+', '+ems[6].string
            save_to_file(result)