clear
Array=csvread('outdir.out.Vm');
t=Array(:, 1);
v=Array(:, 2);
current=Array(:, 3);
figure
subplot(2,1,1)
plot(t,v,'r','LineWidth', 2)
xlim([-100 5000])
ylim([-95 45])
t1=-1000:0.01:0;
hold on
plot(t1,ones(size(t1)) * -70.6,'r','LineWidth', 2)
xlim([-100 5000])
ylim([-95 45])
set(gca,'FontSize',22);
xlabel('time ms')
ylabel('potential diff mV')
subplot(2,1,2)
plot(t,current,'b','LineWidth', 2)
xlim([-100 5000])
hold on
plot(t1,ones(size(t1)) *0,'b','LineWidth', 2)
xlim([-100 5000])
set(gca,'FontSize',22);
xlabel('time ms')
ylabel('current pA')