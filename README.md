# d2l-pytorch-notes

《动手学深度学习》pytorch版本的学习笔记。

地址：https://zh-v2.d2l.ai/

环境配置：[d2l-install](https://zh-v2.d2l.ai/chapter_installation/index.html)



> jupyter notebook 配置

生成密钥

```bash
# 通过ipython生成密码
ipython

In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[3]: 'xxxxxxxxxxxxxxxxxxxxxx'
```



```bash
# 生成配置文件，并添加如下配置
jupyter notebook --generate-config

c.NotebookApp.ip='0.0.0.0'
c.NotebookApp.password = u'xxxxxxxxxxxxxxxxxxxxxx'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888
```



然后就可以通过https://ip:8888远程访问`jupyter notebook`

