---
title: Python 中的 Unicode String 和 Byte String
layout: post
date: 2017-10-27 21:32:38
tags: 
- python
- unicode
categories: 
- python学习笔记
keywords: python,unicode,string
---

# python 2.x 和 python 3.x 字符串类型的区别
python 2.x 中字符编码的坑是历史遗留问题，到 python 3.x 已经得到了很好的解决，在这里简要梳理一下二者处理字符串的思路。
## python 2.x
- `str` 类型：处理 binary 数据和 ASCII 文本数据。
- `unicode` 类型：处理**非 ASCII** 文本数据。

## python 3.x
- `bytes` 类型：处理 binary 数据，同 `str` 类型一样是一个序列类型，其中每个元素均为一个 byte（本质上是一个取值为 0\~255 的整型对象），用于处理二进制文件或数据（如图像，音频等）。
- `str` 类型：处理 unicode 文本数据（包含 ASCII 文本数据）。
- `bytearray` 类型：`bytes` 类型的变种，但是此类型是 **mutable** 的。

---- 

# Unicode 简介
包括 **ASCII 码**、**latin-1 编码** 和 **utf-8 编码** 等在内的码都被认为是 unicode 码。
## 编码和解码的概念
- 编码（encoding）：将字符串映射为一串原始的字节。
- 解码（decoding）：将一串原始的字节翻译成字符串。

## ASCII码
- 编码长度为 1 个 byte.
- 编码范围为 `0x00`\~`0x7F`，只包含一些常见的字符。

## latin-1码
- 编码长度为 1 个 byte.
- 编码范围为 `0x00`\~`0xFF`，能支持更多的字符（如 accent character），兼容 ASCII 码。

## utf-8码
- 编码长度可变，为 1\~4 个 byte。
- 当编码长度为 1 个 byte 时，等同于 ASCII 码，取值为 `0x00` \~ `0x7F`；当编码长度大于 1 个 byte 时，每个 byte 的取值为 `0x80` \~ `0xFF`。

## 其它编码
- utf-16，编码长度为定长 2 个 byte。
- utf-32，编码长度为定长 4 个 byte。

---- 

# Unicode 字符串的存储方式
## 在内存中的存储方式
unicode 字符串中的字符在内存中以一种**与编码方式无关**的方式存储：[unicode code point][1]，它是一个数字，范围为 0\~1,114,111，可以唯一确定一个字符。在表示 unicode 字符串时可以以 unicode code point 的方式表示， 例如在下面的例子 中，`a` 和 `b` 表示的是同一字符串（其中 `'\uNNNN'` 即为 unicode code point，`N` 为一个十六进制位，十六进制位的个数为 4\~6 位；当 unicode code point 的取值在 0\~255 范围内时，也可以 `'\xNN'` 的形式表示）：
```python
# python 2.7
>>> a = u'\u5a1c\u5854\u838e'
>>> b = u'娜塔莎'
>>> print a, b
娜塔莎 娜塔莎
>>> c = u'\xe4'
>>> print c
ä
```
## 在文件等外部媒介中的存储方式
unicode 字符串在文件等外部媒介中须按照指定的编码方式将字符串转换为原始字节串存储。

# 字符表示
## python 3.x
在 python 3.x 中，`str` 类型即可满足日常的字符需求（不论是 ASCII 字符还是国际字符），如下例所示：
```python
# python 3.6
>>> a = 'Natasha, 娜塔莎'
>>> type(a)
str
>>> len(a)
12
```
可以看到，python 3.x 中得到的 `a` 的长度为 12（包含空格），没有任何问题；我们可以对 `a` 进行编码，将其转换为 `bytes` 类型：
```python
# python 3.6
>>> b = a.encode('utf-8')
>>> b
b'Natasha, \xe5\xa8\x9c\xe5\xa1\x94\xe8\x8e\x8e'
>>> type(b)
bytes
```
从上面可以看出，`bytes` 类型的对象中的某个字节的取值在 `0x00` \~ `0x7F` 时，控制台的输出会显示出其对应的 ASCII 码字符，但其本质上是一个原始字节，不应与任何字符等同。
同理，我们也可以将一个 `bytes` 类型的对象译码为一个 `str` 类型的对象：
```python
# python 3.6
>>> a = b.decode('utf-8')
>>> a
'Natasha, 娜塔莎'
```

## python 2.x
在 python 2.x 中，如果还是用 `str` 类型来表示国际字符，就会有问题：
```python 
# python 2.7
>>> a = 'Natasha, 娜塔莎'
>>> type(a)
str
>>> a
'Natasha, \xe5\xa8\x9c\xe5\xa1\x94\xe8\x8e\x8e'
>>> len(a)
18
>>> print a
Natasha, 娜塔莎
```
可以看到，python 2.x 中虽然定义了一个 ASCII 字符和中文字符混合的 `str` 字符串，但实际上 `a` 在内存中存储为一串字节序列，且长度也是该字节序列的长度，很明显与我们的定义字符串的初衷不符合。值得注意的是，这里 `a` 正好是字符串 `'Natasha, 娜塔莎'` 的 utf-8 编码的结果，且将 `a` 打印出来的结果和我们的定义初衷相符合，这其实与控制台的默认编码方式有关，这里控制台的默认编码方式正好是 utf-8，获取控制台的默认编码方式的方式如下:
```python
# python 2.7
>>> import sys
>>> sys.stdin.encoding  # 控制台的输入编码，可解释前例中 a 在内存中的表现形式
'utf-8'
>>> sys.stdout.encoding # 控制台的输出编码，可解释前例中打印 a 的显示结果
'utf-8'
```
另外，`sys.getdefaultencoding()`函数也会得到一种编码方式，得到的结果是系统的默认编码方式，在 python 2.x 中，该函数总是返回 `'ascii'`, 这表明在对字符串编译码时不指定编码方式时所采用的编码方式为ASCII 编码；除此之外，在 python 2.x 中，ASCII 编码方式还会被用作隐式转换，例如 `json.dumps()` 函数在默认情况下总是返回一串字节串，不论输入的数据结构里面的字符串是 unicode 类型还是 str 类型。在 python 3.x 中，隐式转换已经被禁止（也可以说，python 3.x 用不到隐式转换：\>）。
切回正题，在 python 2.x 表示国际字符的正确方式应该是定义一个 `unicode` 类型字符串，如下所示：
```python
# python 2.7
>>> a = u'Natasha, 娜塔莎'
>>> type(a)
unicode
>>> len(a)
12
>>> b = a.encode('utf-8')
>>> b
'Natasha, \xe5\xa8\x9c\xe5\xa1\x94\xe8\x8e\x8e'
>>> type(b)
str
>>> a = b.decode('utf-8')
>>> a
u'Natasha, \u5a1c\u5854\u838e'
```
另外，我们可以对 `unicode` 类型字符串进行编码操作，对 `str` 类型字符串进行译码操作。

---- 

# 文本文件操作
## python 3.x
在 python 3.x 中，文本文件的读写过程中的编解码过程可以通过指定 `open` 函数的参数 `encoding` 的值来自动进行（python 3.x 中的默认情况下文件的编码方式可以由函数 `sys.getfilesystemencoding()`得到，如：
```python
# python 3.6
>>> import sys
>>> sys.getfilesystemencoding()
'utf-8'
>>> a = '娜塔莎'
>>> f = open('data.txt', 'w', encoding='utf-8')
>>> f.write(a)
3
>>> f.close()
>>> f = open('data.txt', 'w', encoding='utf-8')
>>> f.read()
'娜塔莎'
>>> f.close()
```
当然，也可以先手动将字符串编码为字节串，然后再以二进制模式的方式写入文件，再以二进制模式的方式读取文件，最后再手动将读取出来的数据解码为字符串，如：
```python
# python 3.6
>>> a = '娜塔莎'
>>> b = a.encode('utf-8')
>>> f = open('data.txt', 'wb')
>>> f.write(b)
9
>>> f.close()
>>> f.read().decode('utf-8')
'娜塔莎'
>>> f.close()
```
## python 2.x
在 python 2.x 中，`open` 函数只支持读写二进制文件或者文件中的字符大小为 1 个 Byte 的文件，写入的数据为字节，读取出来的数据类型为 `str`；`codecs.open` 函数则支持自动读写 unicode 文本文件，如：
```python
# python 2.7
>>> import codecs
>>> a = u'安德烈'
>>> f = codecs.open('data.txt', 'w', encoding='utf-8')
>>> f.write(a)
>>> f.close()
>>> f = codecs.open('data.txt', 'r', encoding='utf-8') 
>>> print f.read()
安德烈
>>> f.close()
```
类似地，也可以先手动将字符串编码为字节串，然后再以二进制模式的方式写入文件，再以二进制模式的方式读取文件，最后再手动将读取出来的数据解码为字符串，如：
```python
# python 2.7
>>> b = a.encode('utf-8')
>>> f = open('data.txt', 'w')
>>> f.write(b)
>>> f.close()
>>> f = open('data.txt', 'r')
>>> print f.read().decode('utf-8')
安德烈
>>> f.close()
```
总之，在 python 2.x 中读写文件注意两点，一是从文件读取到数据之后的第一件事就是将其按照合适的编码方式译码，二是当所有操作完成需要写入文件时，一定要将要写入的字符串按照合适的编码方式编码。

---- 

# python 2.x 中的 json.dumps() 操作
json 作为一种广为各大平台所采用的数据交换格式，在 python 中更是被广泛使用，然而，在 python 2.x 中，有些地方需要注意。
对于数据结构中的字符串类型为 `str`、 但实际上定义的是一个国际字符串的情况，`json.dumps()` 的结果如下：
```python
# python 2.7
>>> a = {'Natasha': '娜塔莎'}
>>> a_json_1 = json.dumps(a)
>>> a_json_1
'{"Natasha": "\\u5a1c\\u5854\\u838e"}'
>>> a_json_2 = json.dumps(a, ensure_ascii=False)
>>> a_json_2
'{"Natasha": "\xe5\xa8\x9c\xe5\xa1\x94\xe8\x8e\x8e"}'
```
可以看到，在这种情形下，当 `ensure_ascii` 为 `True` 时，`json.dumps()` 操作返回值的类型为 `str`，其会将 `a` 中的中文字符映射为其对应的 unicode code point 的形式，但是却是以 ASCII 字符存储的（即 `'\\u5a1c'` 对应 `6` 个字符而非 `1` 个）；当 `ensure_ascii` 为 `False` 时，`json.dumps()` 操作的返回值类型仍然为 `str`，其会将中文字符映射为其对应的某种 unicode 编码（这里为 utf-8）后的字节串，所以我们将 `a_json_2` 译码就可以得到我们想要的 json：
```python
# python 2.7
>>> a_json_2.decode('utf-8')
u'{"Natasha": "\u5a1c\u5854\u838e"}'
>>> print a_json_2.decode('utf-8')
{"Natasha": "娜塔莎"}
```
对于数据结构中的字符串类型为 `unicode` 的情况，`json.dumps()` 的结果如下：
```python
# python 2.7
>>> u = {u'Natasha': u'娜塔莎'}
>>> u_json_1 = json.dumps(u)
>>> u_json_1
'{"Natasha": "\\u5a1c\\u5854\\u838e"}'
>>> u_json_2 = json.dumps(u, ensure_ascii=False)
>>> u_json_2
u'{"Natasha": "\u5a1c\u5854\u838e"}'
>>> print u_json_2
{"Natasha": "娜塔莎"}
```
在这种情形下，当 `ensure_ascii` 为 `True` 时，`json.dumps()` 操作返回值的类型为 `str`，其得到的结果和前面对 a 操作返回的结果完全一样；而当`ensure_ascii` 为 `False` 时，`json.dumps()` 操作的返回值类型变为 `unicode`，原始数据结构中的中文字符在返回值中完整地保留了下来。
对于数据结构中的字符串类型既有 `unicode` 又有 `str` 的情形，运用 `json.dumps()` 时将 `ensure_ascii` 设为 `False` 的情况又会完全不同。
当数据结构中的 ASCII 字符串为 `str` 类型，国际字符串为 `unicode` 类型时（如 `u = {'Natasha': u'娜塔莎'}`），`json.dumps()` 的返回值是正常的、符合预期的 `unicode` 字符串；
当数据结构中有国际字符串为 `str` 类型，又存在其他字符串为 `unicode` 类型时（如 `u = {u'Natasha': '娜塔莎'}` 或 `u = {u'娜塔莉娅': '娜塔莎'}`），`json.dumps()` 会抛出异常 `UnicodeDecodeError`，这是因为系统会将数据结构中 `str` 类型字符串都转换为 `unicode` 类型，而系统的默认编译码方式为 ascii 编码，因而对 `str` 类型的国际字符串进行 ascii 译码就必然会出错。

[1]:	https://www.wikiwand.com/en/List_of_Unicode_characters