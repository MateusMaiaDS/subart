rm(list=ls())
devtools::load_all()

# Auxiliar functions
sd_new <- function(x){
     sqrt(mean((x - mean(x))^2))
}
standardize <- function(x) (x-mean(x))/sd(x)
ignore_extra <- TRUE


# Loading the data
data_raw <- read.csv("inst/newdata.csv")

data <- dplyr::select(data_raw,
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


x_train <- cbind(data_bsl, data$trt)
colnames(x_train) <- c(colnames(data_bsl), "trt")
x_test <- rbind(x_train, x_train)
x_test$trt <- c(rep(0, nrow(x_train)), rep(1, nrow(x_train)))
y_mat <- as.matrix(data[,c("C","Q")])

n_tree = 100
node_min_size = 5
n_mcmc = 2000
n_burn = 500
alpha = 0.95
beta = 2
nu = 3
sigquant = 0.9
kappa = 2
numcut = 100L # Defining the grid of split rules
usequants = FALSE
m = 20 # Degrees of freed for the classification setting.
varimportance = TRUE
hier_prior_bool = TRUE # Use a hierachical prior or not;
specify_variables = NULL # Specify variables for each dimension (j) by name or index for.
diagnostic = TRUE

y_mat <- y_mat
y_mat[,1] <- ifelse(y_mat[,1]<=mean(y_mat[,1]),1,0)
aux_mod <- subart(x_train = x_train,y_mat = y_mat[,1,drop=FALSE],x_test = x_test,n_tree = 100,n_mcmc = 2000,n_burn = 500)

var_importance_cost <- aux_mod$var_importance[1001:5000,,1] %>% colMeans()
var_importance_qual <- aux_mod$var_importance[1001:5000,,2] %>% colMeans()
names(var_importance_cost) <- colnames(x_train)
names(var_importance_qual) <- colnames(x_train)

sort(var_importance_cost,decreasing = TRUE)
sort(var_importance_qual,decreasing = TRUE)
