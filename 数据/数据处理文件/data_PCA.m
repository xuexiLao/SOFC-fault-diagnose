clear
clc
label_A=ones(5001,1);                          %编号数据类别 正常数据     1
label_B=2*ones(5001,1);                                   % 电堆泄露     2
label_C=3*ones(5001,1);                                   % 鼓风机故障   3
label_D=4*ones(5001,1);                                   % 热交换器故障 4
data_A=readmatrix("F:\个人文档\毕业设计\数据\修改版\正常数据集.xlsx");           %导入数据集，每个数据集样本数量为5000
data_B=readmatrix("F:\个人文档\毕业设计\数据\修改版\电堆泄露.xlsx");
data_C=readmatrix("F:\个人文档\毕业设计\数据\修改版\鼓风机故障.xlsx");
data_D=readmatrix("F:\个人文档\毕业设计\数据\修改版\热交换器机械故障.xlsx");
data_A=data_A(:,2:1:end);                    % 删除时间步（数据集第一列）
data_B=data_B(:,2:1:end);
data_C=data_C(:,2:1:end);
data_D=data_D(:,2:1:end);
data_A=[data_A label_A];                     % 把数据标签嵌入到数据集中
data_B=[data_B label_B];
data_C=[data_C label_C];
data_D=[data_D label_D];
% 重新1000开始取值，避免寻找到异常值，取出11230个样本，10830样本分70%和30%作为训练集和测试集，再用400个样本作为验证集验证
rand_A=randi([1000 4500],[2508 1]);          % 正常数据样本2508
rand_B=randi([1000 4500],[2704 1]);          % 电堆泄露样本2704
rand_C=randi([1000 4500],[3112 1]);          % 鼓风机故障样本3112
rand_D=randi([1000 4500],[2906 1]);          % 热交换器故障样本2906
data_A=data_A(rand_A(:,1),:);                % 打乱数据随机选取
data_B=data_B(rand_B(:,1),:);                % 打乱数据随机选取
data_C=data_C(rand_C(:,1),:);                % 打乱数据随机选取
data_D=data_D(rand_D(:,1),:);                % 打乱数据随机选取
data_total=[data_A;data_B;data_C;data_D];    % 合并选取的正常数据和故障数据集
rand_T=randperm(size(data_total,1))';        % 再次生成乱序，保证数据的随机性
data_total=data_total(rand_T(:,1),:);        % 最终原始数据
data_total_pca=data_total(:,1:11);

%% PCA数据处理
clear;clc
load 数据文件\data_total.mat
load 数据文件\data_total_pca.mat
[n,p]=size(data_total_pca);                      % 获取样本个数和样本特征数
X=zscore(data_total_pca);                        % 对数据标准化
R=corrcoef(X);                                   % 计算协方差矩阵
% R是正半定矩阵，特征值不为负数
% R是对称矩阵，计算时，特征值从小到大排列
[V,D]=eig(R);                                    % V特征向量，D特征值构成的对角矩阵
lambda=diag(D);                                  % 获取矩阵对角线上的值
lambda(lambda<0)=0;                              % 由于精度影响，把很小的负值置为0
lambda=lambda(end:-1:1);                         % 从大到小排列
V=rot90(V)';                                     % 由于特征值从小到大变成了从大到小，特征向量也虚要改变
contribution_rate=lambda/sum(lambda);            % 计算贡献率
sum_contribution_rate=cumsum(lambda)/sum(lambda);% 累计贡献率
%% 主成分数据
F=zeros(size(data_total_pca,1),4);               % 保留4个主成分
for i=1:4
  ai=V(:,i)';                                    % 把第i给特征向量取出，转置为行向量
  Bi=repmat(ai,n,1);                             % 把ai复制n次形成n*p的矩阵
  F(:,i)=sum(Bi.*X,2);                           % 对标准化的数据求了权重后计算每一行的和
end
% 合并最终数据
label=data_total(:,12);
date_finish=[F label];

%% 隐藏部分特征
% clc
% clear
% load F:\matalable文件\毕业设计\数据\PCA处理后数据+特征隐藏后数据\data_finish.mat
% data_finish=date_finish;
% N=[rand(3,11230)*5;rand(1,11230)*1.0]';
% data_finish(:,1:4)=data_finish(:,1:4)+N(:,1:4);



