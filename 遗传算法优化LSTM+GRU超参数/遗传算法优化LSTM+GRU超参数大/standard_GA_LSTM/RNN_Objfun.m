function Obj = RNN_Objfun(X, P, T,  P_test, T_test,NIND,MAXGEN)
%% 用来分别求解种群中各个个体的目标值
%% 输入
% X:种群所有个体的初始权值和阈值
% P:训练样本输入
% T:训练样本输出
% hiddennum:隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
% NIND:种群大小
% MAXGEN:迭代次数
%% 输出
% Obj:所有个体预测样本预测误差的范数
M = size(X,1);
Obj = zeros(M, 1);
for j = 1:M
    fprintf('%d\n',j)                   %显示种群自适应度计算位置
    Obj(j,1) = RNN_Bpfun(X(j, :), P, T, P_test, T_test,NIND,MAXGEN);
end
end