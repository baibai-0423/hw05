# 调试记录

## 问题1：ModuleNotFoundError: No module named 'torch'
- 现象：运行报错找不到torch
- 原因：未安装依赖
- 解决：pip install -r requirements.txt

## 问题2：终端训练日志刷屏
- 现象：训练时输出大量信息
- 原因：批次多，正常输出
- 解决：无需修改，等待完成即可