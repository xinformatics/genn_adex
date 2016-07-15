Array_hh=csvread('compare.out.Vm');
t_hh=Array_hh(:, 1);
v_hh=Array_hh(:,2:21);
figure
plot(t_hh,v_hh)