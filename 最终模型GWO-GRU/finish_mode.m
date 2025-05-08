clc
clear;close all
%% 导入数据，生成乱序数组
load F:\matalable文件\毕业设计\数据\PCA处理后数据+特征隐藏后数据\data2.mat
iris_data=data_finish;         %导入数据
% load F:\matalable文件\毕业设计\数据\验证集数据获取\test_system_data_4.mat
load GWO_GRU_NET97.05.mat

input_train = iris_data(1:7580,1:4)';
output_train = iris_data(1:7580,5)';
input_test = iris_data(7681:7780,1:4)';
output_test = iris_data(7681:7780,5)';
input_verify=iris_data(10831:end,1:4)';
output_verify=iris_data(10831:end,5);

% input_verify2=data_finish(:,1:4)';
% output_verify2=data_finish(:,5);
%% 数据归一化
method=@mapminmax;                 %最大归一化
%method=@mapstd;                   %标准归一化
[input,inputs] = method(input_train);
%[output,outputs] = method(output_train);
input_verify_guiyi = method('apply',input_verify,inputs);
% input_verify2_guiyi = method('apply',input_verify2,inputs);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
input_verify_guiyi  =  double(reshape(input_verify_guiyi , 4, 1, 1, 400));
Input_verify_guiyi=cell(400,1);
for i=1:400
    Input_verify_guiyi{i, 1} = input_verify_guiyi( :, 1, 1, i);
end

% input_verify2_guiyi  =  double(reshape(input_verify2_guiyi , 4, 1, 1, 200));
% Input_verify2_guiyi=cell(200,1);
% for i=1:200
%     Input_verify2_guiyi{i, 1} = input_verify2_guiyi( :, 1, 1, i);
% end

%% 数据预测
t_sim1=predict(net,Input_verify_guiyi);                    %数据后处理
T_sim1=vec2ind(t_sim1')';
accurancy_rate1=sum((T_sim1==output_verify))/400*100;

% t_sim2=predict(net,Input_verify2_guiyi);                    %数据后处理
% T_sim2=vec2ind(t_sim2')';
% accurancy_rate2=sum((T_sim2==output_verify2))/200*100;

%% 排序
[output_verify,N]=sort(output_verify);
TT_sim1=zeros(400,1);
for i=1:400
TT_sim1(i,1)=T_sim1(N(i,1));
end

% [output_verify2,N]=sort(output_verify2);
% TT_sim2=zeros(200,1);
% for i=1:200
% TT_sim2(i,1)=T_sim2(N(i,1));
% end
%% 绘图
figure
plot(1:400,output_verify,'r-*',1:400,TT_sim1,'b-*');
xlabel('样本序号');
ylabel('样本类别');
legend('真实值','预测值');

% figure
% plot(1:200,output_verify2,'r-*',1:200,TT_sim2,'b-*');
% xlabel('样本序号');
% ylabel('样本类别');
% legend('真实值','预测值');

%% 混淆矩阵的绘制
figure
cm=confusionchart(output_verify,TT_sim1);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'verifythe mode of SOFC fault diagnose';

