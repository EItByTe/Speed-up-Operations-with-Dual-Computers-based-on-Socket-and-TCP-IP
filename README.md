# 基于 Socket 和 TCP/IP 协议实现双机多线程协作运算加速
## Speed-up-Operations-with-Dual-Computers-based-on-Socket-and-TCP-IP

通过算法、多进程、分布式计算加速简单运算、排序等操作。

包括`普通算法`, `openmp`, `SSE`, `手写多线程`, `归并排序算法`，`cuda显卡加速求和`等方式运算速度耗时对比, 于pdf文件中。
### Client
客户端，通过TCP/IP协议将数据发送给主机端进行协作计算。


### Server
主机端，通过TCP/IP协议接受Client端发来的数据，进行加速计算。
