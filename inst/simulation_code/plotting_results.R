# Plotting results
library(tidyverse)
rm(list=ls())
devtools::load_all()
set.seed(42)
n_ <- 1000
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 2
task_ <- "regression" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman2" # For this can be either 'friedman1' or 'friedman2'

result <- readRDS(paste0("~/spline_bart_lab/mvnbart6/inst/cv_data/regression/result/",
                         sim_,"_",task_,"_n_",n_,"_p_",p_,"_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))


library(cowplot)
# Plotting first the RMSE for the test sample
result_df <- do.call(rbind,result)
rmse_plot <- result_df %>% filter(metric == "rmse_test") %>%
     mutate(mvn_dim = as.factor(mvn_dim)) %>%
     ggplot()+
     geom_boxplot(mapping = aes(x = model, y = value))+
     ylab("RMSE test")+
     xlab("Model")+
     facet_wrap(~mvn_dim,scales = "free_y")+
     theme_classic()

crps_plot <- result_df %>% filter(metric == "crps_test") %>%
     mutate(mvn_dim = as.factor(mvn_dim)) %>%
     ggplot()+
     geom_boxplot(mapping = aes(x = model, y = value))+
     ylab("CRPS test")+
     xlab("Model")+
     facet_wrap(~mvn_dim,scales = "free_y")+
     theme_classic()


pi_plot <- result_df %>% filter(metric == "pi_test") %>%
     mutate(mvn_dim = as.factor(mvn_dim)) %>%
     ggplot()+
     geom_boxplot(mapping = aes(x = model, y = value))+
     geom_hline(yintercept = 0.5, lty = 'dashed', col = 'blue')+
     ylab("PI coverage")+
     xlab("Model")+
     facet_wrap(~mvn_dim,scales = "free_y")+
     theme_classic()

cowplot::plot_grid(rmse_plot,crps_plot,pi_plot,ncol = 3)

