# -*- coding: utf-8 -*-
# @Time        : 2019/3/8 11:45
# @Author      : tianyunzqs
# @Description : 

import os
import sys
import re
import json

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.string_utils import strQ2B


cls = {
    "sports": "运动",
    "stock": "股票",
    "business": "商业",
    "yule": "娱乐",
    "it": "IT",
    "learning": "教育",
    "daxue.learning": "教育",
    "baobao": "宝宝",
    "fund": "财政",
    "goabroad": "出国"
}


def get_data(path):
    def get_cls(url):
        for c in cls:
            if c in url:
                return cls[c]
        return None

    data = []
    re_doc = re.compile("^<doc>$")
    re_url = re.compile("^<url>.*</url>$")
    re_docno = re.compile("^<docno>.*</docno>$")
    re_contenttitle = re.compile("^<contenttitle>.*</contenttitle>$")
    re_content = re.compile("^<content>.*</content>$")
    re_doc2 = re.compile("^</doc>$")
    with open(path, "r", encoding='ANSI') as f:
        lines = f.readlines()
        d = {}
        for line in tqdm(lines):
            line = line.strip().replace(r"\u3000", u"　")
            if re_doc.search(line):
                d = {}
            elif re_url.search(line):
                line = line.replace("<url>", "").replace("</url>", "")
                line = line[:line.find(r".sohu.com")]
                classify = get_cls(line)
                if classify:
                    d["classify"] = classify
            elif re_docno.search(line):
                continue
            elif re_contenttitle.search(line) and d and d["classify"]:
                d["title"] = strQ2B(line.replace("<contenttitle>", "").replace("</contenttitle>", ""))
            elif re_content.search(line) and d and d["classify"]:
                d["content"] = strQ2B(line.replace("<content>", "").replace("</content>", ""))
            elif re_doc2.search(line) and d:
                data.append(d)
                d = {}
    return data


if __name__ == '__main__':
    result = get_data(path=r"D:\alg_file\data\news_sohusite_xml.dat")
    with open(r"D:\alg_file\data\data.dat", "w", encoding='utf-8') as f:
        for res in result:
            f.write(json.dumps(res, ensure_ascii=False))
            f.write("\n")
