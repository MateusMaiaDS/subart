arma::mat residuals_ = y_mat_hat-z_mat_train;

arma::mat D_sqrt = sqrt(data.D);
data.W = D_sqrt*data.R*D_sqrt;
arma::mat W_proposal = arma::iwishrnd(m*data.W,m);
arma::mat D_proposal = arma::diagmat(W_proposal);
arma::mat inv_D_sqrt  = arma::inv(sqrt(D_proposal));
arma::mat R_proposal = inv_D_sqrt*W_proposal*inv_D_sqrt;

double alpha_corr = exp(log_posterior_dens(R_proposal,D_proposal,nu,residuals_,false) - log_posterior_dens(data.R,data.D,nu,residuals_,false) + log_proposal_dens(data.R,data.D,nu,R_proposal,D_proposal,m) - log_proposal_dens(R_proposal,D_proposal,nu,data.R,data.D,m));
if(arma::randu(arma::distr_param(0.0,1.0)) < alpha_corr) {
     data.R = R_proposal;
     data.D = D_proposal;
     data.Sigma = data.R;
}  else {
     data.Sigma = data.R;
}
