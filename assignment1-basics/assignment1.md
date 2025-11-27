# Assignment1 作业回答

## 2.1 unicode1 Understanding Unicode

(a) `chr(0)`返回`\x00`，是Unicode中的空字符

(b) `print(chr(0).__repr__())`会打印出chr(0)的字符串表达，即`\x00`，直接`print(chr(0))`打印为空

(c) "this is a test" + chr(0) + "string" 会返回'this is a test\x00string'

print("this is a test" + chr(0) + "string") 会打印出this is a teststring

所以当`chr(0)`出现在文本时不可见，不会打印出来

## 2.2 unicode2 Unicode Encodings

(a) UTF-8是变长(1~4字节)编码且兼容ASCII码，而UTF-16变长(2/4字节)不兼容ASCII，UTF32固定4字节不兼容ASCII。使用UTF-8节省空间且广泛使用

(b) 该函数把每个字节分别进行解码，这只对编码成单字节的英文有效，如果是中文/表情这类的就不行

(c) `print(bytes([0b10000000]).decode("utf-8"))`不符合utf8编码规则，单字节的utf8必须以二进制最高位0开始，即`print(bytes([0b01111111]).decode("utf-8"))`可以