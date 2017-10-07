function acc = evaluation(label,Y)
t_test = sum(label == Y);
all_test = length(Y);
acc = t_test/all_test;
end