alpha = 0.8;
tau1 = 0.00045;
tau2 = 0.0007;
time_steps = 20000;
time_steps_vector = 0:1:time_steps;

P_h1 = alpha *exp(-tau1 * time_steps_vector);
P_h2 = alpha *exp(-tau2 * time_steps_vector);




plot(time_steps_vector,P_h1, 'LineWidth',2, 'Color','r')
hold on 

plot(time_steps_vector,P_h2, 'LineWidth',2, 'Color','b')

legend('tau = 0.00045','tau = 0.0007')


title('Exponential decay')
xlabel('Time steps')
ylabel('% of feedback')

xlim([0 time_steps])


xlabel('Accumulated_time')
ylabel('Ph')

xlim([0 time_steps])