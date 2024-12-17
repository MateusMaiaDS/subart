# Plotting results
library(tidyverse)
rm(list=ls())
devtools::load_all()
setwd("/users/research/mmarques/spline_bart_lab/spline_bart_lab/mvnbart6/")
set.seed(42)
n_ <- 250
p_ <- 10
n_tree_ <- 100
mvn_dim_ <- 2
seed_ <- 43
task_ <- "regression" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'

# Setting the path where all results are
models <- if (task_ == "classification") {
     c("bayesSUR","subart","bart")
} else if (task_ == "regression" & mvn_dim_==2){
     c("bayesSUR","subart","bart","mvBART")
} else if (task_ == "regression" & mvn_dim_==3){
     c("bayesSUR","subart","bart")
} else {
     stop("No valid setting")
}

# Getting the directory path where all results are stored (Change depending where you running this)
results_path <- paste0("/users/research/mmarques/r1_rebuttal_results/")

# Generating the result df that will be stored
result_df <- data.frame()
result_df_corr <- data.frame()

#Last round of results
for(model in models){
     result <- readRDS(paste0(results_path,"seed_",seed_,"_",model,"_",sim_,"_",task_,"_n_",n_,"_p_",p_,"_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))
     result_df <- rbind(result_df, lapply(result,function(x){x$comparison_metrics}) %>% do.call(rbind,.) )
     result_df_corr <- rbind(result_df_corr, lapply(result,function(x){x$correlation_metrics}) %>% do.call(rbind,.))
}


library(cowplot)

# Changing the factor
if(task_=="regression"){
     result_df <- result_df %>%
          mutate(model = ifelse(model=="bayesSUR","BayesSUR",model)) %>%
          mutate(model = factor(model,levels = c("suBART","BART", "mvBART","BayesSUR"))) %>%
          mutate(mvn_dim = ifelse(mvn_dim==1,"j=1",mvn_dim)) %>%
          mutate(mvn_dim = ifelse(mvn_dim==2,"j=2",mvn_dim)) %>%
          mutate(mvn_dim = ifelse(mvn_dim==3,"j=3",mvn_dim)) %>%
          mutate(mvn_dim = factor(mvn_dim, levels = c("j=1","j=2","j=3")))


     if(mvn_dim_==2) {
          text_size <- 16
     } else if(mvn_dim_ == 3){
          text_size <- 16
     }

     rmse_plot <- result_df %>% filter(metric == "rmse_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = (value),col = model),show.legend = FALSE)+
          # scale_y_log10(labels = scales::number_format(accuracy = 0.01))+ # Maybe use the log-scale
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
          scale_color_manual(values = c("suBART" = "#F8766D",
                                        "BART" = "#00BA38",
                                        "mvBART" = "#C77CFF",
                                        "BayesSUR" = "#619CFF"))+
          ylab("RMSE")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size))

     # Just a decoy to get a legend
     rmse_plot_legend <- result_df %>% filter(metric == "rmse_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = (value),col = model),show.legend = TRUE)+
          # scale_y_log10(labels = scales::number_format(accuracy = 0.01))+# Maybe use the log-scale
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
          scale_color_manual(values = c("suBART" = "#F8766D",
                                        "BART" = "#00BA38",
                                        "mvBART" = "#C77CFF",
                                        "BayesSUR" = "#619CFF"))+
          ylab("RMSE")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size),legend.position = 'bottom')

     rmse_plot_legend <- rmse_plot_legend + labs(color = "Model:")

     crps_plot <- result_df %>% filter(metric == "crps_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = (value),col = model),show.legend = FALSE)+
          # scale_y_log10(labels = scales::number_format(accuracy = 0.01))+# Maybe use the log-scale
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
          scale_color_manual(values = c("suBART" = "#F8766D",
                                        "BART" = "#00BA38",
                                        "mvBART" = "#C77CFF",
                                        "BayesSUR" = "#619CFF"))+
          ylab("CRPS")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size))

     pi_range <- result_df %>% dplyr::group_by(metric) %>% summarise(min = min(value), max = max(value)) %>% filter(metric=='pi_test') %>% select(min,max) %>% c %>% unlist()
     # result_df[model=="BayesSUR",metric:=ifelse(metric=="ci_test","pi_test",metric)]

     pi_plot <- result_df %>% filter(metric == "pi_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = value,col = model),show.legend = FALSE)+
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+ #
          geom_hline(yintercept = 0.5, lty = 'dashed', col = 'black')+
          scale_color_manual(values = c("suBART" = "#F8766D",
                                        "BART" = "#00BA38",
                                        "mvBART" = "#C77CFF",
                                        "BayesSUR" = "#619CFF"))+
          ylim(pi_range)+
          ylab("PI coverage")+
          xlab("")+
          facet_wrap(~mvn_dim)+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                panel.spacing.x = unit(2.0,'lines'),
                text = element_text(size = text_size))

     legend <- cowplot::get_plot_component(rmse_plot_legend,pattern = 'guide-box-bottom',return_all = TRUE )

     main_plot <-  cowplot::plot_grid(cowplot::plot_grid(rmse_plot,crps_plot,pi_plot,ncol = 3),
                                      legend,nrow = 2,
                                      rel_heights = c(1, 0.1))  # Adjust legend width as needed

     main_plot

     # Set fixed width and height for the plot
     if(mvn_dim_==2){
          width_inches <- 15  # Adjust as needed
          height_inches <- 5  # Adjust as needed
     } else if(mvn_dim_==3){
          width_inches <- 15  # Adjust as needed
          height_inches <- 5  # Adjust as needed
     } else {
          stop("Enter valid mvn_dim")
     }

     # # Export as TIFF
     # # (attention change your path here)
     # ggsave(paste0("/localusers/researchers/mmarques/spline_bart_lab/mvbart6_plots_skew/may2024_",sim_,"_",task_,"_",n_,"_",mvn_dim_,".tiff"), plot = main_plot,
     #        width = width_inches, height = height_inches, dpi = 300)
     #
     # # Export as PDF
     # ggsave(paste0("/localusers/researchers/mmarques/spline_bart_lab/mvbart6_plots_skew/FIG_4_september_new_2024",sim_,"_",task_,"_",n_,"_",mvn_dim_,".pdf"), plot = main_plot,
     #        width = width_inches, height = height_inches)
} # End of iteration

# Plotting results for the CR coverage
summary_df_corr_cov <-result_df_corr %>% filter(metric == "cr_cov") %>%  group_by(param_index,model) %>%
     summarise(mean_cv = mean(value)) %>% pivot_wider(names_from = model, values_from = mean_cv) %>%
     mutate(metric = "PI coverage")

summary_df_corr_rmse <- result_df_corr %>% filter(metric == "rmse") %>%  group_by(param_index,model) %>%
     summarise(mean_cv = sqrt(mean(value^2))) %>% pivot_wider(names_from = model, values_from = mean_cv) %>%
     mutate(metric = "RMSE")

summary_corr_comparison <- rbind(summary_df_corr_cov,
                                 summary_df_corr_rmse) %>%
     dplyr::filter(stringr::str_detect(param_index,"sigma"))

summary_corr_comparison

