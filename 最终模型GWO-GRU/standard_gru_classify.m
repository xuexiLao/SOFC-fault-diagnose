clc
clear;close all
%% 导入数据，生成乱序数组·
load F:\matalable文件\毕业设计\数据\PCA处理后数据+特征隐藏后数据\verify_data2.mat
iris_data=data_finish;         %导入数据
% arry=randperm(10830);
%% 划分数据
input_train = iris_data(1:7581,1:4)';
output_train = iris_data(1:7581,5)';
input_test = iris_data(7582:10830,1:4)';
output_test = iris_data(7582:10830,5)';

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

numFeatures=size(input,1);
layers1 = [
    sequenceInputLayer(numFeatures)
    gruLayer(122,'OutputMode','last','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')       
    % LSTM层，64个隐藏单元 'sequence'输出一个序列，'last'输出最后一个时间步'sequence'输出一个序列，'last'输出最后一个时间步,对输入权重和循环权重采用HE初始化方法，有效避免梯度爆炸
    reluLayer                              %reluLayer激活函数，避免梯度爆炸
    fullyConnectedLayer(4)                 % 全连接层 对于输出结果的大小，对于分类任务后面一般需要加一个softmax层和分类层
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
 'MaxEpochs',50, ...
 'MiniBatchSize',161,...
 'GradientThreshold',1, ...               %梯度阈值用于防止梯度爆炸问题。当梯度超过这个阈值时，会被截断，通常为1-5
 'InitialLearnRate',0.0381, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',20, ...
 'LearnRateDropFactor',0.2, ...
 'L2Regularization',0.0001,...              %正则化因子
 'Verbose',false, ...
 'Plots','training-progress'...
    );

net = trainNetwork(Input,output,layers1,options);

t_sim1=predict(net,Input);                    %数据后处理
t_sim2=predict(net,Input_test_guiyi);
T_sim1=vec2ind(t_sim1')';
T_sim2=vec2ind(t_sim2')';
accurancy_rate1=sum((T_sim1==output_train'))/7581*100;
accurancy_rate2=sum((T_sim2==output_test'))/3249*100;

%% 排序
[output_train,N]=sort(output_train);
TT_sim1=zeros(7581,1);
for i=1:7581
TT_sim1(i,1)=T_sim1(N(1,i));
end

[output_test,I]=sort(output_test);
TT_sim2=zeros(3249,1);
for i=1:3249
TT_sim2(i,1)=T_sim2(I(1,i));
end
%% 绘图
figure
plot(1:7581,output_train,'r-*',1:7581,TT_sim1,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('真实值','预测值');

figure
plot(1:3249,output_test,'r-*',1:3249,TT_sim2,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('真实值','预测值');

 save("GWO_GRU_NET3","net")