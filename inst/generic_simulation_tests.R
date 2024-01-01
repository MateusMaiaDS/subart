# Generating generic simulations to see if it is appropriate.
rm(list=ls())
devtools::load_all()
set.seed(42)
n_ <- 250
p_ <-  10
n_tree_ <- 50
mvn_dim_ <- 3

# Simulating for the scenario one
train_sample <- sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
test_sample <- sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)


mvn_bart <- mvnbart(x_train = train_sample$x,
                    y_mat = train_sample$y,
                    x_test = test_sample$x,n_tree = n_tree_,
                    n_mcmc = 2500,n_burn = 1000,varimportance = TRUE,
                    df = 10)
# For BART it is necessary to iterate over each (j)
bart_mod <- vector("list",mvn_dim_)
for(i in 1:mvn_dim_){
     bart_mod[[i]] <- dbarts::bart(x.train = train_sample$x,
                          y.train = train_sample$y[,i],
                          x.test = test_sample$x,ntree = n_tree_,
                          ndpost = 1500,nskip = 1000)
}

j_ <- 3
rmse(mvn_bart$y_hat_test_mean[,j_],test_sample$y[,j_])
rmse(bart_mod[[j_]]$yhat.test.mean,test_sample$y[,j_])

