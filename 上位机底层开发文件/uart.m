s = serialport("COM9",9600,"Timeout",10);



data = read(s,15,"string");
date=fgetl(s)
data2=uint8(data)
disp(data)
fclose(s)
delete(s)