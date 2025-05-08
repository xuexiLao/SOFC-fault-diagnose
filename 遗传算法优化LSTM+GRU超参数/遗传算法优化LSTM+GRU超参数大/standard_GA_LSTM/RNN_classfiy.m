clc
clear;close all

%% 导入数据，生成乱序数组
iris_data=readmatrix('F:\matalable文件\CNN\iris\iris_data.xlsx');         %导入数据
arry=randperm(150);
%% 划分数据
input_train = iris_data(arry(1:100),1:4)';
output_train = iris_data(arry(1:100),5)';
input_test = iris_data(arry(101:150),1:4)';
output_test = iris_data(arry(101:150),5)';
% 节点个数
input_num = 4;                    %4个特征
hidden_num = 7;                   %隐藏层神经元
output_num = 3;                   %输出类别
%% 把输出1维转换为3维[1 0 0] [0 1 0] [0 0 1]
output=zeros(100,3);
for i=1:100
    switch output_train(i)
        case 1
        output(i,:)=[1 0 0];
        case 2
        output(i,:)=[0 1 0];
        case 3
        output(i,:)=[0 0 1];
    end
end
[xx,output]=max(output,[],2);
% % output=zeros(3,100);
% % for i=1:100
% %     switch output_train(i)
% %         case 1
% %         output(:,i)=[1 0 0]';
% %         case 2
% %         output(:,i)=[0 1 0]';
% %         case 3
% %         output(:,i)=[0 0 1]';
% %     end
% % end

output=categorical(output);
output2=categorical(output_test);
% for i=1:100
% Output{i,1}=output(i,:);
% end
%可以使用ind2vec函数
%% 数据归一化
method=@mapminmax;                 %最大归一化
%method=@mapstd;                    %标准归一化
[input,inputs] = method(input_train);
%[output,outputs] = method(output_train);
input_test_guiyi = method('apply',input_test,inputs);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
input =  double(reshape(input, 4, 1, 1, 100));
input_test_guiyi  =  double(reshape(input_test_guiyi , 4, 1, 1, 50));

for i = 1 : 100
    Input{i, 1} = input(:, 1, 1, i);
end
for i=1:50
    Input_test_guiyi{i, 1} = input_test_guiyi( :, 1, 1, i);
end

numFeatures=size(input,1);
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(6,'OutputMode','last','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')       
    % LSTM层，64个隐藏单元 'sequence'输出一个序列，'last'输出最后一个时间步'sequence'输出一个序列，'last'输出最后一个时间步,对输入权重和循环权重采用HE初始化方法，有效避免梯度爆炸
    reluLayer                              %reluLayer激活函数，避免梯度爆炸
    fullyConnectedLayer(3)                 % 全连接层 对于输出结果的大小，对于分类任务后面一般需要加一个softmax层和分类层
    softmaxLayer
    classificationLayer
];

% layers = [ ...
%   sequenceInputLayer(4)               % 输入层
%   
%   lstmLayer(64, 'OutputMode', 'last')   % LSTM层
%   reluLayer                             % Relu激活层
%   
%   fullyConnectedLayer(3)                % 全连接层
%   softmaxLayer                          % 分类层
%   classificationLayer];

options = trainingOptions('adam', ...
 'MaxEpochs',1000, ...
 'GradientThreshold',1, ...               %梯度阈值用于防止梯度爆炸问题。当梯度超过这个阈值时，会被截断，通常为1-5
 'InitialLearnRate',0.005, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',750, ...
 'LearnRateDropFactor',0.2, ...
 'Verbose',false, ...
 'Plots','training-progress'...
    );
% options = trainingOptions('adam', ...       % Adam 梯度下降算法
%     'MaxEpochs', 1000, ...                  % 最大迭代次数
%     'InitialLearnRate', 0.01, ...           % 初始学习率
%     'LearnRateSchedule', 'piecewise', ...   % 学习率下降
%     'LearnRateDropFactor', 0.1, ...         % 学习率下降因子
%     'LearnRateDropPeriod', 750, ...         % 经过 750 次训练后 学习率为 0.01 * 0.1
%     'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
%     'ValidationPatience', Inf, ...          % 关闭验证
%     'Plots', 'training-progress', ...       % 画出曲线
%     'Verbose', false);

net = trainNetwork(Input,output,layers,options);

t_sim1=predict(net,Input);                    %数据后处理
t_sim2=predict(net,Input_test_guiyi);
T_sim1=vec2ind(t_sim1')';
T_sim2=vec2ind(t_sim2')';
accurancy_rate1=sum((T_sim1'==output_train))/100*100;
accurancy_rate2=sum((T_sim2'==output_test))/50*100;

analyzeNetwork(net);
lstmLayer = net.Layers(2);                     %获取网络超参数
inputweight=lstmLayer.InputWeights;            %获取输入权重
recurrentweight=lstmLayer.RecurrentWeights;    %获取循环权重
bia=lstmLayer.Bias;                            %获取偏差