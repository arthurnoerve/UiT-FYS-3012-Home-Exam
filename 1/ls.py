X_train, X_test = split_data(data,r)
Y_train, Y_test = split_data(labels,r)

(w_lms, c_lms) = widrow_hoff(X_train,Y_train,min_rho)
hit_lms = linear_test(w_lms,X_test,Y_test)

w_ls = least_squares(X_train,Y_train)
hit_ls = linear_test(w_ls,X_test,Y_test)
