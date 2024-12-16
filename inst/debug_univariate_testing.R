# Loading the model and testing in the univariate case
devtools::load_all()
library(dbarts)
set.seed(42)

# Generating the univariate sample
j <- 1
n <- 250
n_tree <- 50

# Generating the simulation friedman1
sim_data <- subart::sim_mvn_friedman1(n = 250,p = 10,mvn_dim = 2)
x_train <- sim_data$x
y_train <- sim_data$y[,j,drop=FALSE]
true_sigma <- sim_data$Sigma[j,j]

# Adjusting a subart model
subart_mod <- subart(x_train = x_train,y_mat = y_train,x_test = x_train,n_tree = 50)
