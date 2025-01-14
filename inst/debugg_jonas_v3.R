# install.packages("devtools")
# devtools::install_github("MateusMaiaDS/subart", ref = "feat/cat_predictor_class")
# packages ####
devtools::load_all()

library(subart)
library(BART)
library(mvtnorm)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(skewBART)
library(surbayes)
library(progress)
# functions
sd_new <- function(x){
     sqrt(mean((x - mean(x))^2))
}
standardize <- function(x) (x-mean(x))/sd(x)
ignore_extra <- TRUE

# clean data ####
data_raw <- read.csv("inst/newdata.csv")



data <- select(data_raw,
               "Groep",
               "TOTALHEALTHCARE",
               "EQ5DNLSCORE_T3",
               "Leeftijd",
               "Geslacht",
               "OPLEIDING",
               "VoorgeschiedenisCAT",
               "Trauma01",
               "Letsels",
               "ISS",
               "Opnameduur",
               "Opname",
               "Operatie",
               "TijdTraumaPoli"
)
data <- dplyr::rename(data,
                      trt = Groep,
                      Q = EQ5DNLSCORE_T3,
                      C = TOTALHEALTHCARE,
                      gender = Geslacht,
                      age = Leeftijd,
                      education = OPLEIDING,
                      disease_history = VoorgeschiedenisCAT,
                      trauma_type = Trauma01,
                      fracture_region = Letsels,
                      hospital_duration = Opnameduur,
                      surgery = Operatie,
                      admission = Opname,
                      TTO = TijdTraumaPoli
)
data$trt <- as.integer(as.factor(data$trt)) - 1
data$gender <- as.factor(data$gender)
#data$education <- as.factor(data$education) # keep this as numeric?
data$disease_history <- as.factor(data$disease_history)
data$trauma_type <- as.factor(data$trauma_type)
data$fracture_region <- as.factor(data$fracture_region)
data$surgery <- as.factor(data$surgery)

data_bsl <- data.frame(
     gender = as.integer(data$gender) - 1,
     age = standardize(data$age),
     education = data$education,
     disease_history = data$disease_history,
     trauma_type = data$trauma_type,
     fracture_region = data$fracture_region,
     ISS = standardize(data$ISS),
     hospital_duration = data$hospital_duration,
     surgery = as.integer(data$surgery) - 1,
     TTO = standardize(data$TTO),
     admission = as.integer(as.factor((data$admission)))-1
)
# define simulation ####

## prognostic functions and treatment effects ####
# include: age, disease history, ISS, TTO
mu_C <- function(gender, age, disease_history, education, TTO, surgery, ISS){
     return(2000 + 500 * age + (-200)*education + 500 * surgery)
}

tau_C <- function(gender, age, disease_history, education, TTO, surgery, ISS){
     return(500)
}

mu_Q <- function(gender, age, disease_history, education, TTO, surgery, ISS){
     return(0.5 + 0.2*sin(age)*(gender + 1))
}
tau_Q <- function(gender, age, disease_history, education, TTO, surgery, ISS){
     return(-0.1 + 0.1*exp(-TTO))
}

## create dataset with true potential outcomes and treatment effects ####
n <- nrow(data_bsl)
data_true <- data.frame(tau_C = rep(NA, n))
data_true$mu_C <- NA
data_true$tau_Q <- NA
data_true$mu_Q <- NA

for (i in 1:n){
     data_true$mu_C[i] <- mu_C(gender = data_bsl$gender[i],age =  data_bsl$age[i], disease_history = data_bsl$disease_history[i],education = data_bsl$education[i], TTO = data_bsl$TTO[i], surgery = data_bsl$surgery[i], ISS = data_bsl$ISS[i])
     data_true$tau_C[i] <- tau_C(gender = data_bsl$gender[i],age =  data_bsl$age[i], disease_history = data_bsl$disease_history[i],education = data_bsl$education[i], TTO = data_bsl$TTO[i], surgery = data_bsl$surgery[i], ISS = data_bsl$ISS[i])
     data_true$mu_Q[i] <- mu_Q(gender = data_bsl$gender[i],age =  data_bsl$age[i], disease_history = data_bsl$disease_history[i],education = data_bsl$education[i], TTO = data_bsl$TTO[i], surgery = data_bsl$surgery[i], ISS = data_bsl$ISS[i])
     data_true$tau_Q[i] <- tau_Q(gender = data_bsl$gender[i],age =  data_bsl$age[i], disease_history = data_bsl$disease_history[i],education = data_bsl$education[i], TTO = data_bsl$TTO[i], surgery = data_bsl$surgery[i], ISS = data_bsl$ISS[i])
}
data_true$ps <- 0.9 * pnorm(-0.5 + 1*data_bsl$surgery -1.5 * standardize(data_true$mu_Q)) + 0.05
plot(data_true$mu_Q, data_true$ps)
Delta_C_true <- mean(data_true$tau_C)
Delta_Q_true <- mean(data_true$tau_Q)
INB_20_true <- 20000 * Delta_Q_true - Delta_C_true
INB_50_true <- 50000 * Delta_Q_true - Delta_C_true
plot(data_true$ps, data_true$tau_Q)

## make simulation function ####
sim <- function(n,data_true,rho){
     data_sim <- data.frame(trt = rep(NA, n),
                            C = rep(NA, n),
                            Q = rep(NA, n)
     )
     data_sim$trt <- rbinom(n, 1, data_true$ps)
     sigma_C <- 500
     sigma_Q <- 0.05
     Sigma <- matrix(c(sigma_C^2, sigma_C * sigma_Q * rho, sigma_C * sigma_Q * rho, sigma_Q^2), ncol = 2)
     epsilon <- mvtnorm::rmvnorm(n, c(0,0), Sigma)
     data_sim$C <- data_true$mu_C + data_sim$trt * data_true$tau_C + epsilon[,1]
     data_sim$Q <- data_true$mu_Q + data_sim$trt * data_true$tau_Q + epsilon[,2]

     return(data_sim)
}

# make simulated datasets ####
n_sim <-  1000
rho <- -0.25
data_sim_list <- list()
for (i in 1:n_sim){
     data_sim_list[[i]] <- sim(data_true = data_true,n = nrow(data_true), rho = rho)
}

# testing new probit bart ####
data_sim <- data_sim_list[[1]]
X_train <- cbind(data_bsl, data_sim$trt)
colnames(X_train) <- c(colnames(data_bsl), "trt")
X_test <- rbind(X_train, X_train)
X_test$trt <- c(rep(0, nrow(X_train)), rep(1, nrow(X_train)))
Y_train <- as.matrix(data_sim[,c("C","Q")])
n_mcmc <- 2000
n_burn <- 500
ps_fit <- gbart(x.train = select(X_train, !c("trt")),
                y.train = as.matrix(X_train$trt),
                type = "pbart",
                ndpost = n_mcmc,
                nskip = n_burn,
                ntree = 1,
                keepevery = 1)
X_train$ps <- apply(pnorm(ps_fit$yhat.train), 2, mean)
X_test <- rbind(X_train, X_train)
X_test$trt <- c(rep(0, nrow(X_train)), rep(1, nrow(X_train)))
ps_fit_new <- subart(x_train = select(X_train, !c("trt")),
                     y_mat = as.matrix(X_train$trt),
                     x_test = select(X_train, !c("trt")),
                     n_mcmc = n_burn + n_mcmc,
                     n_burn = n_burn,
                     n_tree = 100
)




