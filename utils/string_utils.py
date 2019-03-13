# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 16:53
# @Author      : tianyunzqs
# @Description : 


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


if __name__ == '__main__':
    print(strQ2B("ＧＰＳｍｎ123abc博客园ｓｐｐｘｉａｏ＠ｓｏｈｕ新科　ＨＦ５８（２ＧＢ）"))
