mvnBart_wrapper <- function(x_train, x_test, c_train, q_train, rescale){

     if (rescale == TRUE){

          c_rescaled <- (c_train - min(c_train))/(max(c_train) - min(c_train)) - 0.5
          q_rescaled <- (q_train - min(q_train))/(max(q_train) - min(q_train)) - 0.5

          mvnBart_fit <- mvnbart::mvnbart(x_train = x_train,
                                          c_train = c_rescaled, q_train = q_rescaled,
                                          x_test = x_test, scale_bool = FALSE,
                                          n_tree = 200)

          c_hat_test <- (mvnBart_fit$c_hat_test + 0.5) * (max(c_train) - min(c_train)) + min(c_train)
          q_hat_test <- (mvnBart_fit$q_hat_test + 0.5) * (max(q_train) - min(q_train)) + min(q_train)

          # something is wrong with the following backtransformations
          sigma_c <- (max(c_train) - min(c_train)) * mvnBart_fit$tau_c_post^(-1/2)
          sigma_q <- (max(q_train) - min(q_train)) * mvnBart_fit$tau_c_post^(-1/2)
          rho = mvnBart_fit$rho_post

     }

     else{
          mvnBart_fit <- mvnbart::mvnbart(x_train = x_train,
                                          c_train = c_train, q_train = q_train,
                                          x_test = x_test, scale_bool = FALSE,
                                          n_tree = 200)

          # something is wrong with the following backtransformations
          sigma_c <- (max(c_train) - min(c_train)) * mvnBart_fit$tau_c_post^(-1/2)
          sigma_q <- (max(q_train) - min(q_train)) * mvnBart_fit$tau_c_post^(-1/2)
          rho = mvnBart_fit$rho_post
     }

     return(
          list(
               c_hat_test = mvnBart_fit$c_hat_test,
               q_hat_test = mvnBart_fit$q_hat_test,
               sigma_c = mvnBart_fit$tau_c_post^(-1/2),
               sigma_q = mvnBart_fit$tau_q_post^(-1/2),
               rho = mvnBart_fit$rho_post
          )
     )
}

fit <- mvnBart_wrapper(data_train[,1:4], data_test[,1:4], data_train$C, data_train$Q, rescale = F)

1/N * sum((apply(fit$c_hat_test, 1, mean) - data_test$C)) # out of sample bias - c
1/N * sum((apply(fit$q_hat_test, 1, mean) - data_test$Q)) # out of sample bias - q
sqrt(1/N * sum((apply(fit$c_hat_test, 1, mean) - data_test$C)^2)) # out of sample RMSE - c
sqrt(1/N * sum((apply(fit$q_hat_test, 1, mean) - data_test$Q)^2)) # out of sample RMSE - q

summary(fit$sigma_c)
summary(fit$sigma_q)
summary(fit$rho)

# fit seperate models ####
univariate.fit.C <- wbart(x.train = as.matrix(data_train[,1:4]), y.train = as.matrix(data_train$C),
                          x.test = as.matrix(data_test[,1:4]), ntree = 200,
                          ndpost=1500, nskip=500)

univariate.fit.Q <- wbart(x.train = as.matrix(data_train[,1:4]), y.train = as.matrix(data_train$Q),
                          x.test = as.matrix(data_test[,1:4]), ntree = 200,
                          ndpost=1500, nskip=500)


1/N * sum(univariate.fit.C$yhat.test.mean - data_test$C) # out of sample bias - c
1/N * sum(univariate.fit.Q$yhat.test.mean - data_test$Q) # out of sample bias - q
sqrt(1/N * sum((univariate.fit.C$yhat.test.mean - data_test$C)^2)) # out of sample RMSE - c
sqrt(1/N * sum((univariate.fit.Q$yhat.test.mean - data_test$Q)^2)) # out of sample RMSE - q

