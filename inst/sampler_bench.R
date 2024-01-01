rm(list=ls())
library(microbenchmark)
devtools::load_all()
library(purrr)
n_ <- 1000
mean_ <- -5
microbenchmark::microbenchmark( pkg_impl <- msm::rtnorm(n = n_,mean = mean_,sd = 1, lower = 0),
                                cpp_impl <- replicate(n = n_,truncated_sample(mu = mean_, TRUE)) )

min(pkg_impl)
min(cpp_impl)
density(pkg_impl) %>% plot(col = 'red')
density(cpp_impl) %>% lines(col = "blue")


