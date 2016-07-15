Array_hh=csvread('compare.out.Vm');
t_hh=Array_hh(:, 1);
v_hh=Array_hh(:,2:21);
d=8;
figure
plot(t_hh,v_hh(:,d))
xlabel('Time (ms)')
ylabel('Voltage (mV)')
set(gca,'box','off')
set(gcf,'PaperPositionMode','auto');
pos=get(gcf,'pos');
set(gcf,'PaperSize',[pos(3), pos(4)]);
% print(gcf,'Mbody1_KC_11.jpeg','-djpeg','-r600');