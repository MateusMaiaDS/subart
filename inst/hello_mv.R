rm(list=ls())
library(tidyverse)
devtools::load_all()
require(tgp)
require(SoftBart)
n_ <- 200
set.seed(42)

# Simulation 1
sd_ <- 0.1
fried_sim <- mlbench::mlbench.friedman1(n = n_,sd = sd_)
friedman_no_interaction <- function (n, sd = 1)
{
        x <- matrix((runif(5 * n)), ncol = 5)
        # x <- x[order(x[,1]),]
        # y <- 10 * sin(pi * x[, 1] )
        # y <- y + 20 * (x[, 2] - 0.5)^2 + 10 * x[, 3] + 5 * x[, 4]
        y <- 20 * (x[, 1] - 0.5)^2

        if (sd > 0) {
                y <- y + rnorm(n, sd = sd)
        }
        list(x = as.data.frame(x), y = y)
}

fried_sim_new_sample <- mlbench::mlbench.friedman1(n = n_+1,sd = sd_)

# Getting the df's
x <- fried_sim$x %>% as.data.frame()
        # dplyr::mutate(cat_var = as.factor(sample(c("A","B","C"),size = n_,replace = TRUE)))

x_test <- fried_sim_new_sample$x %>% as.data.frame()
        #dplyr::mutate(cat_var = as.factor(sample(c("A","B","C"),size = n_+1,replace = TRUE)))

y <- fried_sim$y

# # Getting new predictions values
# x <- matrix(sort(runif(n = n_,min = -pi,max = pi))) %>% as.data.frame()
# x_test <- matrix(sort(runif(n = n_,min = -pi,max = pi))) %>% as.data.frame()
# colnames(x) <- colnames(x_test) <- "x"
# y <- sin(x) + rnorm(10,sd = sd_)


# # Testing the gpbart2
# bart_test <- gpbart(x_train = x,y = unlist(c(y)),x_test = x_test,
#                    n_tree = 10,n_mcmc = 2000,alpha = 0.5,
#                    beta = 2,
#                    df = 3,sigquant = 0.9,stump = FALSE,
#                    n_burn = 500,scale_bool = TRUE)
bart_test <- bart2(x_train = x,y = unlist(c(y)),x_test = x_test,
                                            n_tree = 200,n_mcmc = 2000,alpha = 0.95,node_min_size = 1,
                                            beta = 2,
                                            df = 3,sigquant = 0.9,stump = FALSE,
                                            n_burn = 500,scale_bool = FALSE)
bart_test$y_hat %>% rowMeans()

low_quantile <- bart_test$y_hat_test %>% apply(1,function(x){quantile(x,probs = 0.25)})
up_quantile <- bart_test$y_hat_test %>% apply(1,function(x){quantile(x,probs = 0.75)})
#
# plot(x$x,bart_test$y_hat %>% rowMeans(),pch = 20)
# lines(x_test$x,up_quantile, pch= 20, col = "orange", lty = "dashed")
# lines(x_test$x,low_quantile, pch= 20, col = "orange", lty = "dashed")
# lines(x_test$x,bart_test$y_hat_test %>% rowMeans(), pch= 20, col = "blue")
# lines(x_test$x,bart_test$y_hat_test %>% rowMeans()+1.96*bart_test$y_test_sd_post %>% rowMeans(), pch = 20, col = "blue", lty = "dashed")
# lines(x_test$x,bart_test$y_hat_test %>% rowMeans()-1.96*bart_test$y_test_sd_post %>% rowMeans(), pch = 20, col = "blue", lty = "dashed")

# bart_test_without_phi
#
# gp_bart_test <- gpbart::gp_bart(x_train = x,y_train = unlist(c(y)),n_mcmc = 2500,n_burn = 500,
#                                 x_test = x_test,n_tree = 1,gp_variables_ = colnames(x))

comparison <- microbenchmark::microbenchmark(my_bart <- bart3::bart2(x_train = x,y = unlist(c(y)),x_test = x_test,
                                                                n_tree = 50,n_mcmc = 2500,alpha = 0.95,
                                                                beta = 2,tau = diff(range(y)),node_min_size = 1,
                                                                df = 3,sigquant = 0.9,stump = FALSE,
                                                                n_burn = 500,scale_bool = TRUE),
                                             bart_mod <- dbarts::bart(x.train = x,y.train = unlist(c(y)),ntree = 50,x.test = x_test,keeptrees = TRUE),
                                             times = 1)

# # Running BART
bartmod <- dbarts::bart(x.train = x,y.train = unlist(c(y)),ntree = 200,x.test = x_test,keeptrees = TRUE)

# Convergence plots
par(mfrow = c(1,2))
plot(bart_test$tau_post,type = "l", main = expression(tau),ylab=  "")
plot(bartmod$sigma^-2, type = "l", main = paste0("BART: ",expression(tau)),ylab=  "")

par(mfrow = c(1,2))
plot(bart_test$y_hat %>% rowMeans(),unlist(c(y)), main = 'gpbart2', xlab = "gpbart2 pred", ylab = "y")
plot(bartmod$yhat.train.mean,unlist(c(y)), main = "BART", xlab = "BART pred", ylab = "y")
par(mfrow=c(1,1))
plot(bart_test$y_hat %>% rowMeans(),bartmod$yhat.train.mean)
my_train_pred <- bart_test$y_hat %>% rowMeans()
dbarts_pred <- bartmod$yhat.train.mean

lm_mod <- lm(my_train_pred ~ dbarts_pred)
lm_mod %>% summary
# Comparing on the test set
pred_bart <- colMeans(predict(bartmod,fried_sim_new_sample$x %>% as.data.frame()))

# Plotting the 2d-data;
par(mfrow = c(1,1))
plot(x$x,y$x)
lines(x$x,bart_test$y_hat %>% rowMeans(), col = "blue")
lines(x$x,bartmod$yhat.train.mean , col = "red")
