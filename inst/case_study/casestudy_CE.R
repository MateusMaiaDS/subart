library(mice)
# library(mvnbart6)
devtools::load_all()
library(bayesplot)
library(dplyr)
data_raw <- read.csv("inst/case_study/20200610_OLVG_DD_database_Economic_final.csv")

data_raw <- select(data_raw,
                   "Def_Inclusion",
                   "EQ5D_Eco",
                   "HealthCare_Costs_Eco",
                   "Age",
                   contains("numeric")
)

data_imp <- mice(data_raw, m = 1)
data_imp <- complete(data_imp, action = "long", include = F)

X_train <- select(data_imp,
                  "Def_Inclusion",
                  "Age",
                  contains("numeric")
)

X_test <- rbind(X_train, X_train)
X_test$Def_Inclusion <- c(rep(0, nrow(X_train)), rep(1, nrow(X_train)))

Y_train <- select(data_imp,
                  "EQ5D_Eco",
                  "HealthCare_Costs_Eco",
)

mvbart_fit <- mvnbart(x_train = X_train,
                      y_mat = as.matrix(Y_train),
                      x_test = X_test,
                      n_tree = 100,
                      n_mcmc = 100,
                      n_burn = 0,
                      df = 10)

mvnbart_fit$ATE <- apply()
post_df <- data.frame(
  sigma1 = sqrt(mvbart_fit$Sigma_post[1,1,]),
  sigma2 = sqrt(mvbart_fit$Sigma_post[2,2,]),
  rho = mvbart_fit$Sigma_post[1,2,]/(sqrt(mvbart_fit$Sigma_post[1,1,] * mvbart_fit$Sigma_post[2,2,])),
  y1_hat = mvbart_fit$y_hat[1,1,],
  y2_hat = mvbart_fit$y_hat[1,2,]
)
mcmc_trace(post_df)
ESS(post_df)
