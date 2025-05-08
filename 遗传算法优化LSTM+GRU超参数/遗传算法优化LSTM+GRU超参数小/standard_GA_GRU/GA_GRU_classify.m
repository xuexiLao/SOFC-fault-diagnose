clc
clear;close all
clear functions
%% 导入数据，生成乱序数组
load E:\matlab_document\毕业设计\数据\PCA处理后数据+特征隐藏后数据\data.mat
iris_data=data_finish;         %导入数据
arry=randperm(10830);
%% 划分数据 
input_train = iris_data(arry(1:7581),1:4)';
output_train = iris_data(arry(1:7581),5)';
input_test = iris_data(arry(7582:end),1:4)';
output_test = iris_data(arry(7582:end),5)';

% 节点个数
input_num = 4;                    %4个特征
hidden_num = 64;                   %隐藏层神经元
output_num = 4;                   %输出类别
%% 把输出1维转换为4维[1 0 0 0] [0 1 0 0] [0 0 1 0] [0 0 0 1]
output=zeros(7581,4);
for i=1:7581
    switch output_train(i)
        case 1
        output(i,:)=[1 0 0 0];
        case 2
        output(i,:)=[0 1 0 0];
        case 3
        output(i,:)=[0 0 1 0];
        case 4
        output(i,:)=[0 0 0 1];
    end
end
[xx,output]=max(output,[],2);

output=categorical(output);
output2=categorical(output_test);
%可以使用ind2vec函数
%% 数据归一化
method=@mapminmax;                 %最大归一化
%method=@mapstd;                   %标准归一化
[input,inputs] = method(input_train);
%[output,outputs] = method(output_train);
input_test_guiyi = method('apply',input_test,inputs);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
input =  double(reshape(input, 4, 1, 1, 7581));
input_test_guiyi  =  double(reshape(input_test_guiyi , 4, 1, 1, 3249));
Input=cell(7581,1);
Input_test_guiyi=cell(3249,1);
for i = 1 : 7581
    Input{i, 1} = input(:, 1, 1, i);
end
for i=1:3249
    Input_test_guiyi{i, 1} = input_test_guiyi( :, 1, 1, i);
end

%% 区域描述器参数
optimize_num=4;                 %优化个数
Precision=[20 20 20 20];        %每个参数的二进制位数
LB=[1e-4 1e-4 10 32];           %参数下界Lower bound
UB=[0.1 0.1 128 128];           %参数上界Upper bound
coding=[1;0;1;1];               %编码方式
%% 定义遗传算法参数
NIND = 42;                      % 种群大小
MAXGEN = 10;                    % 最大遗传代数
PRECI = 20;                     % 个体长度(变量二进制位数）
GGAP = 0.95 ;                   % 代沟
px = 0.7;                       % 交叉概率
pm = 0.05;                       % 变异概率
trace = zeros(optimize_num + 1, MAXGEN);   % 寻优结果的初始值
FieldD = [repmat(Precision, 1, 1); repmat([LB; UB], 1, 1); repmat(coding, 1, optimize_num)]; % 区域描述器
Chrom = crtbp(NIND, PRECI * optimize_num); % 创建任意离散随机种群
% 优化
gen = 0;                                                % 代计数器
X = bs2rv(Chrom, FieldD);                               % 计算初始种群的十进制转换
ObjV = RNN_Objfun(X, Input, output, Input_test_guiyi, output_test);      % 计算目标函数值(训练输入，训练输出，隐藏神经元，测试输入，测试输出）
% LSTM和GRU训练输入的结果为categrical,预测输入只需输入类别
%% 取出精确率和召回率
idex_pr=zeros(1,MAXGEN);                                           %每代最优索引
position_total=zeros(MAXGEN*(MAXGEN-2),optimize_num+1);            %取出所有种群所有代中的位置及fitness

%%
while gen < MAXGEN
    fprintf('遗传代数：%d\n', gen)
    FitnV = ranking(ObjV);                              % 分配适应度值
    SelCh = select('sus', Chrom, FitnV, GGAP);          % 随机采样选择
    SelCh = recombin('xovmp', SelCh, px);               % 多点交叉xovmp,单点交叉xovsp重组
    SelCh = mut(SelCh, pm);                             % 变异
    X = bs2rv(SelCh, FieldD);                           % 子代个体的二进制到十进制转换
    ObjVSel = RNN_Objfun(X, Input, output, Input_test_guiyi, output_test);    % 计算子代的目标函数值
    [Chrom, ObjV] = reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel);                             % 将子代重插入到父代，得到新种群
    X = bs2rv(Chrom, FieldD);                           % 子代个体的二进制到十进制转换
    gen = gen + 1;                                      % 代计数器增加
    % 获取每代的最优解及其序号，Y为最优解，I为个体的序号
    [Y, I] = min(ObjV);
    idex_pr(1,gen)=I;%取出每代值最优索引
    % position_total((gen:(NIND-2))*gen,1:4)=X(1:(NIND-2),:);%获取特征
    % position_total((gen:(NIND-2))*gen,5)=ObjVSel(:,1);%获取fitness
    trace(1: optimize_num, gen) = X(I, :);              % 记下每代的最优值
    trace(end, gen) = Y;                                % 记下每代的最优值
    
    % 对变异和重组值进行动态调整
    std_ObjVSel=std(ObjVSel);
    if std_ObjVSel<1e-2
       px = 0.8;pm = 0.1;
    else
       px = 0.7;pm = 0.05;
    end
end
