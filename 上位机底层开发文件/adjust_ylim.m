function [y_down,y_up] = adjust_ylim(data)
% 解决y轴坐标滑动
% data:输入数据
ylim_max=max(data);
ylim_min=min(data);
gap=(ylim_max-ylim_min)*0.2;
y_down=ylim_min-gap;
y_up=ylim_max+gap;

end