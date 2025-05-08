function updateTimeHandler(app)
    % 获取当前时间
    currentDateTime = datetime('now', 'TimeZone', 'Asia/Shanghai');
    % 更新标签文本
    app.beijing_time.Text = string(currentDateTime);

end