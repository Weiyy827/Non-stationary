%% The function that generates the sequence M
function [seq]=mseq(coef)
%***************************************************
% 此函数用来生成m序列
% coef为反馈系数向量
% Author: FastestSnail
% Date: 2017-10-03
%***************************************************
m=length(coef);
len=2^m-1; % 得到最终生成的m序列的长度     
backQ=0; % 对应寄存器运算后的值，放在第一个寄存器
seq=zeros(1,len); % 给生成的m序列预分配
registers = [1 zeros(1, m-2) 1]; % 给寄存器分配初始结果
for i=1:len
    seq(i)=registers(m);
    backQ = mod(sum(coef.*registers) , 2); %特定寄存器的值进行异或运算，即相加后模2
    registers(2:length(registers)) = registers(1:length(registers)-1); % 移位
    registers(1)=backQ; % 把异或的值放在第一个寄存器的位置
end
end