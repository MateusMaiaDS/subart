#include "mvnbart6.h"
#include <iomanip>
#include<cmath>
#include <random>
#include <RcppArmadillo.h>
using namespace std;


// =====================================
// Statistics Function
// =====================================

arma::mat generateZ(int n, int d, const arma::mat& Sigma_true_chol) {
        int i, j;

        // Generate an n x d matrix to store the result
        arma::mat Z(n, d);

        for (i = 0; i < n; ++i) {
                // Generate a d-dimensional random vector
                arma::vec rnormVec = arma::randn(d);

                // Multiply by the Cholesky decomposition
                for (j = 0; j < d; ++j) {
                        Z(i, j) = arma::dot(Sigma_true_chol.row(j), rnormVec);
                }
        }

        return Z;
}


// Function to create a correlation matrix from correlation coefficients
//[[Rcpp::export]]
arma::mat makeSigma(arma::vec sigma, int d) {

        // Checking the inputs
        if(sigma.size()!= (d*(d-1))/2){
                Rcpp::stop("No match among sigma vector and d-dimension");
        }

        arma::mat Sigma(d, d, arma::fill::eye);

        int index = 0;
        for (int i = 0; i < d; i++) {
                for (int j = i + 1; j < d; j++) {
                        Sigma(i, j) = sigma(index);
                        Sigma(j, i) = sigma(index);
                        index++;
                }
        }

        return Sigma;
}

// [[Rcpp::export]]
arma::vec makeSigmaInv(arma::mat& Sigma) {
        int d = Sigma.n_rows;
        arma::vec sigma((d * (d - 1)) / 2);

        int k = 0;
        for (int j = 0; j < d; j++) {
                for (int i = j + 1; i < d; i++) {
                        sigma(k++) = Sigma(i, j);
                }
        }

        return sigma;
}




// I don't need to calculate this, they cancell out in the MH hasting
double multivariate_gamma(double nu, int p) {
        double result = 0.0;
        for (int i = 1; i <= p; ++i) {
                result += (lgamma((nu + 1 - i) / 2));
        }
        result = exp(result);
        // cout << "Is this stable??  "<< result << endl;
        return std::pow(arma::datum::pi, p * (p - 1) / 4) * result;
}

// Function to calculate the log-likelihood of Wishart distribution
// [[Rcpp::export]]
double wishart_loglikelihood(const arma::mat& X, const arma::mat& Sigma, double nu) {
        int p = X.n_rows;  // Dimensionality
        double logdet_X = arma::log_det(X).real();  // Log determinant of X
        double logdet_Sigma = arma::log_det(Sigma).real();  // Log determinant of Sigma
        double trace_invXSigma = arma::trace(arma::inv(Sigma) * X);

        double log_likelihood = 0.5*(nu - p - 1)  * logdet_X - 0.5 * trace_invXSigma - (nu * 0.5) * logdet_Sigma - (nu * p * 0.5 ) * log(2.0) - log(multivariate_gamma(nu,p));

        return log_likelihood;
}

// [[Rcpp::export]]
double iwishart_loglikelihood(const arma::mat& X, const arma::mat& Sigma, double nu) {
        int p = X.n_rows;  // Dimensionality
        double logdet_X = arma::log_det(X).real();  // Log determinant of X
        double logdet_Sigma = arma::log_det(Sigma).real();  // Log determinant of Sigma
        double trace_inv_Sigma_invX = arma::trace(Sigma * arma::inv(X));


        // THE FULL VERSION HAS A PROBLEM TO COMPUTE THE MULTIVARIATE GAMMA FUNCTION WHEN m is large, investigate that later
        // double log_likelihood = -0.5*(nu + p + 1)  * logdet_X - 0.5 * trace_inv_Sigma_invX + (nu * 0.5) * logdet_Sigma - (nu * p * 0.5 ) * log(2.0) - log(multivariate_gamma(nu,p));
        double log_likelihood = -0.5*(nu + p + 1)  * logdet_X - 0.5 * trace_inv_Sigma_invX + (nu * 0.5) * logdet_Sigma - (nu * p * 0.5 ) * log(2.0);// - log(multivariate_gamma(nu,p));

        return log_likelihood;
}

double log_prior_dens(const arma::mat & R, const arma::mat & D, double nu){
        unsigned int d = D.n_cols;
        arma::mat sqrt_D = sqrt(D);
        // cout << "Sqrt_D" << sqrt_D << endl;
        arma::mat W = sqrt_D*R*sqrt_D;
        // cout << "Wishhart density ::" << iwishart_loglikelihood(W,arma::eye(d,d),nu+d-1) << endl;
        // cout << "Second term: " << ((d-1)*0.5)*sum(log(D.diag())) <<endl;
        return iwishart_loglikelihood(W,arma::eye(d,d),nu+d-1) +  ((d-1)*0.5)*sum(log(D.diag()));
}
double log_posterior_dens(const arma::mat & R, const arma::mat & D, double nu,
                          const arma::mat & Z, bool sample_prior){
        if(sample_prior){
                return log_prior_dens(R,D,nu);
        } else {
                double n_ = Z.n_rows;
                double log_prior_dummy = 0.0;
                log_prior_dummy = log_prior_dens(R,D,nu)-0.5*n_*arma::log_det(R).real();
                arma::mat inv_R = arma::inv(R);
                // arma::cout << "invR dimensions: " << inv_R.n_rows << " x " << inv_R.n_cols;
                for(unsigned int i =0 ; i < Z.n_rows ; i++) {
                        log_prior_dummy = log_prior_dummy - 0.5*arma::as_scalar(Z.row(i)*inv_R*Z.row(i).t());
                }
                return log_prior_dummy;
        }
}

double log_proposal_dens(const arma::mat & R_star, const arma::mat & D_star, double nu,
                         const arma::mat& R, const arma::mat& D, int m) {

        arma::mat sqrt_D_star = sqrt(D_star);
        arma::mat sqrt_D  = sqrt(D);
        arma::mat W_star = sqrt_D_star*R_star*sqrt_D_star;
        arma::mat W = sqrt_D*R*sqrt_D;
        // cout << "W VALUE:: "<< W << endl;
        // cout << "W STAR VALUE:: "<< W_star << endl;

        double  d = D.n_cols;
        // cout << "PROPOSAL DENS DEBUGG: " << iwishart_loglikelihood(W_star, m * W, m) << endl;
        return iwishart_loglikelihood(W_star, m * W, m) + (0.5*(d-1))*sum(log(D_star.diag()));
}


// Function to calculate the log of a MVN distribution
//[[Rcpp::export]]
double log_dmvn(arma::vec& x, arma::mat& Sigma){

        arma::mat L = arma::chol(Sigma ,"lower"); // Remove diagonal later
        arma::vec D = L.diag();
        double p = Sigma.n_cols;

        arma::vec z(p);
        double out;
        double acc;

        for(int ip=0;ip<p;ip++){
                acc = 0.0;
                for(int ii = 0; ii < ip; ii++){
                        acc += z(ii)*L(ip,ii);
                }
                z(ip) = (x(ip)-acc)/D(ip);
        }
        out = (-0.5*sum(square(z))-( (p/2.0)*log(2.0*M_PI) +sum(log(D)) ));


        return out;

};


// //[[Rcpp::export]]
arma::mat sum_exclude_col(arma::mat mat, int exclude_int){

        // Setting the sum matrix
        arma::mat m(mat.n_rows,1);

        if(exclude_int==0){
                m = sum(mat.cols(1,mat.n_cols-1),1);
        } else if(exclude_int == (mat.n_cols-1)){
                m = sum(mat.cols(0,mat.n_cols-2),1);
        } else {
                m = arma::sum(mat.cols(0,exclude_int-1),1) + arma::sum(mat.cols(exclude_int+1,mat.n_cols-1),1);
        }

        return m;
}



// Initialising the model Param
modelParam::modelParam(arma::mat x_train_,
                        arma::mat y_mat_,
                        arma::mat x_test_,
                        arma::mat x_cut_,
                        int n_tree_,
                        int node_min_size_,
                        double alpha_,
                        double beta_,
                        double nu_,
                        arma::vec sigma_mu_,
                        arma::mat Sigma_,
                        arma::mat S_0_wish_,
                        arma::vec A_j_vec_,
                        double n_mcmc_,
                        double n_burn_,
                        bool sv_bool_,
                        arma:: mat sv_matrix_,
                        arma::vec categorical_indicators_){


        // Assign the variables
        x_train = x_train_;
        y_mat = y_mat_;
        x_test = x_test_;
        xcut = x_cut_;
        n_tree = n_tree_;
        node_min_size = node_min_size_;
        alpha = alpha_;
        beta = beta_;
        nu = nu_;
        sigma_mu = sigma_mu_;

        Sigma = Sigma_;
        S_0_wish = S_0_wish_;
        A_j_vec = A_j_vec_;
        a_j_vec = arma::vec(y_mat_.n_cols,arma::fill::zeros);
        n_mcmc = n_mcmc_;
        n_burn = n_burn_;

        sv_bool = sv_bool_;
        sv_matrix = sv_matrix_;

        // Generating the elements for the correlation matrix
        R = Sigma_;
        D = arma::eye(y_mat_.n_cols,y_mat_.n_cols);
        // Grow acceptation ratio
        move_proposal = arma::vec(3,arma::fill::zeros);
        move_acceptance = arma::vec(3,arma::fill::zeros);

        categorical_indicators = categorical_indicators_;
        categorical_indicators_bool = sum(categorical_indicators_)==0 ? false: true;

}

// Initialising a node
Node::Node(modelParam &data){
        isLeaf = true;
        isRoot = true;
        left = NULL;
        right = NULL;
        parent = NULL;
        train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
        test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1) ;

        var_split = -1;
        var_split_rule = -1.0;
        lower = 0.0;
        upper = 1.0;
        mu = 0.0;
        n_leaf = 0.0;
        n_leaf_test = 0;
        log_likelihood = 0.0;
        depth = 0;


}

Node::~Node() {
        if(!isLeaf) {
                delete left;
                delete right;
        }
}

// Initializing a stump
void Node::Stump(modelParam& data){

        // Changing the left parent and right nodes;
        left = this;
        right = this;
        parent = this;
        // n_leaf  = data.x_train.n_rows;

        // Updating the training index with the current observations
        for(int i=0; i<data.x_train.n_rows;i++){
                train_index[i] = i;
        }

        // Updating the same for the test observations
        for(int i=0; i<data.x_test.n_rows;i++){
                test_index[i] = i;
        }

}

void Node::addingLeaves(modelParam& data){

     // Create the two new nodes
     left = new Node(data); // Creating a new vector object to the
     right = new Node(data);
     isLeaf = false;

     // Modifying the left node
     left -> isRoot = false;
     left -> isLeaf = true;
     left -> left = left;
     left -> right = left;
     left -> parent = this;
     left -> var_split = 0;
     left -> var_split_rule = -1.0;
     left -> lower = 0.0;
     left -> upper = 1.0;
     left -> mu = 0.0;
     left -> log_likelihood = 0.0;
     left -> n_leaf = 0.0;
     left -> depth = depth+1;
     left -> train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
     left -> test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1);

     right -> isRoot = false;
     right -> isLeaf = true;
     right -> left = right; // Recall that you are saving the address of the right node.
     right -> right = right;
     right -> parent = this;
     right -> var_split = 0;
     right -> var_split_rule = -1.0;
     right -> lower = 0.0;
     right -> upper = 1.0;
     right -> mu = 0.0;
     right -> log_likelihood = 0.0;
     right -> n_leaf = 0.0;
     right -> depth = depth+1;
     right -> train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
     right -> test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1);


     return;

}

// Creating boolean to check if the vector is left or right
bool Node::isLeft(){
        return (this == this->parent->left);
}

bool Node::isRight(){
        return (this == this->parent->right);
}

// This functions will get and update the current limits for this current variable
void Node::getLimits(){

        // Creating  a new pointer for the current node
        Node* x = this;
        // Already defined this -- no?
        lower = 0.0;
        upper = 1.0;
        // First we gonna check if the current node is a root or not
        bool tree_iter = x->isRoot ? false: true;
        while(tree_iter){
                bool is_left = x->isLeft(); // This gonna check if the current node is left or not
                x = x->parent; // Always getting the parent of the parent
                tree_iter = x->isRoot ? false : true; // To stop the while
                if(x->var_split == var_split){
                        tree_iter = false ; // This stop is necessary otherwise we would go up til the root, since we are always update there is no prob.
                        if(is_left){
                                upper = x->var_split_rule;
                                lower = x->lower;
                        } else {
                                upper = x->upper;
                                lower = x->var_split_rule;
                        }
                }
        }
}





void Node::deletingLeaves(){

     // Should I create some warn to avoid memoery leak
     //something like it will only delete from a nog?
     // Deleting
     delete left; // This release the memory from the left point
     delete right; // This release the memory from the right point
     left = this;  // The new pointer for the left become the node itself
     right = this; // The new pointer for the right become the node itself
     isLeaf = true;

     return;

}
// Getting the leaves (this is the function that gonna do the recursion the
//                      function below is the one that gonna initialise it)
void get_leaves(Node* x,  std::vector<Node*> &leaves_vec) {

        if(x->isLeaf){
                leaves_vec.push_back(x);
        } else {
                get_leaves(x->left, leaves_vec);
                get_leaves(x->right,leaves_vec);
        }

        return;

}



// Initialising a vector of nodes in a standard way
std::vector<Node*> leaves(Node* x) {
        std::vector<Node*> leaves_init(0); // Initialising a vector of a vector of pointers of nodes of size zero
        get_leaves(x,leaves_init);
        return(leaves_init);
}

// Sweeping the trees looking for nogs
void get_nogs(std::vector<Node*>& nogs, Node* node){
        if(!node->isLeaf){
                bool bool_left_is_leaf = node->left->isLeaf;
                bool bool_right_is_leaf = node->right->isLeaf;

                // Checking if the current one is a NOGs
                if(bool_left_is_leaf && bool_right_is_leaf){
                        nogs.push_back(node);
                } else { // Keep looking for other NOGs
                        get_nogs(nogs, node->left);
                        get_nogs(nogs, node->right);
                }
        }
}



// Collect the Split VARS
void collect_split_vars(arma::vec& var_count, Node* tree){

        // Iterating over all the terminal nodes
        if(!tree->isLeaf){
                var_count[tree->var_split] = var_count[tree->var_split] + 1;
                collect_split_vars(var_count, tree->left);
                collect_split_vars(var_count,tree->right);
        }

}
// Creating the vectors of nogs
std::vector<Node*> nogs(Node* tree){
        std::vector<Node*> nogs_init(0);
        get_nogs(nogs_init,tree);
        return nogs_init;
}



// Initializing the forest
Forest::Forest(modelParam& data){

        // Creatina vector of size of number of trees
        trees.resize(data.n_tree*data.y_mat.n_cols);
        for(int  i=0;i<(data.n_tree*data.y_mat.n_cols);i++){
                // Creating the stump for each tree
                trees[i] = new Node(data);
                // Filling up each stump for each tree
                trees[i]->Stump(data);
        }
}


// Selecting a random node
Node* sample_node(std::vector<Node*> leaves_){

        // Getting the number of leaves
        int n_leaves = leaves_.size();
        // return(leaves_[std::rand()%n_leaves]);
        if((n_leaves == 0) || (n_leaves==1) ){
             return leaves_[0];
        } else {
             return(leaves_[arma::randi(arma::distr_param(0,(n_leaves-1)))]);
        }

}

// [[Rcpp::export]]
arma::mat replaceWithUniqueRankMatrix(arma::vec categories, arma::vec values) {
     // Step 1: Group values by categories
     std::map<double, std::vector<double>> valueGroups;
     for (size_t i = 0; i < categories.n_elem; ++i) {
          valueGroups[categories[i]].push_back(values[i]);
     }

     // Step 2: Compute the mean for each group (based on `values`)
     std::vector<std::pair<double, double>> means; // Pair: (category, mean)
     for (const auto& [category, group_values] : valueGroups) {
          double sum = std::accumulate(group_values.begin(), group_values.end(), 0.0);
          double mean = sum / group_values.size();
          means.emplace_back(category, mean);
     }

     // Step 3: Rank categories based on strictly increasing means
     std::sort(means.begin(), means.end(),
               [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                    return a.second < b.second; // Sort by mean
               });

     // Assign strict ranks (1-based)
     std::map<double, int> ranks;
     int rank = 1;
     for (const auto& [category, mean] : means) {
          ranks[category] = rank++;
     }

     // Step 4: Create a matrix for unique values and ranks
     arma::mat result(ranks.size(), 2); // Rows = unique categories, Columns = 2
     int row = 0;
     for (const auto& [category, rank] : ranks) {
          result(row, 0) = category; // Unique category
          result(row, 1) = rank;     // Corresponding rank
          row++;
     }

     return result;
}

arma::vec replaceWithMatrixMapping(arma::vec inputVector, arma::mat mappingMatrix) {
     arma::vec outputVector = inputVector; // Copy input vector to modify

     // Iterate through the input vector
     for (size_t i = 0; i < inputVector.n_elem; ++i) {
          double originalValue = inputVector[i];
          // Find the corresponding value in the mappingMatrix
          for (size_t j = 0; j < mappingMatrix.n_rows; ++j) {
               if (mappingMatrix(j, 0) == originalValue) {
                    // Replace the value with the new mapped value
                    outputVector[i] = mappingMatrix(j, 1);
                    break;
               }
          }
     }

     return outputVector;
}


// Grow a tree for a given rule
void grow(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec& curr_u, int y_j_){

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* g_node = sample_node(t_nodes);

        // Store all old quantities that will be used or not
        int old_var_split = g_node->var_split;
        double old_var_split_rule = g_node->var_split_rule;

        // // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Error gpNodeLogLike" << endl;
                t_nodes[i]->updateResiduals(data, curr_res,curr_u); // Do I need to do this?
        }

        // Calculating the likelihood of the current grown node
        g_node->nodeLogLike(data);
        // cout << "LogLike Node ok Grow" << endl;

        // Adding the leaves
        g_node->addingLeaves(data);

        bool no_valid_node = false;
        int p_try = 0;
        // arma::cout << "variable (j): " << y_j_ << endl;
        int sum_vars = arma::sum(data.sv_matrix.row(y_j_));
        arma::vec split_candidates;

        // Trying to find a cutpoint
        if(data.sv_bool){
                arma::vec split_candidates_aux(sum_vars);
                int index_aux_ = 0;
                for(int ii_ = 0; ii_ < data.sv_matrix.n_cols; ii_++) {
                        if(data.sv_matrix(y_j_,ii_)==1){
                                split_candidates_aux(index_aux_) = ii_;
                                index_aux_++;
                        }
                }
                split_candidates = arma::shuffle(split_candidates_aux);

        } else {
                arma::vec split_candidates_aux = arma::shuffle(arma::regspace(0,1,data.x_train.n_cols-1));
                split_candidates = split_candidates_aux;
        }

        Rcpp::NumericVector valid_cutpoint;
        arma::mat recoded_ranks;

        while(!no_valid_node){
             g_node->var_split = split_candidates(p_try);

             Rcpp::NumericVector var_split_range;


             if(data.categorical_indicators(g_node->var_split)==0) {
                  // Getting the maximum and the minimum;
                  for(int i = 0; i < g_node->n_leaf; i++){
                       var_split_range.push_back(data.x_train(g_node->train_index[i],g_node->var_split));
                  }

                  // Getting the minimum and the maximum;
                  double max_rule = max(var_split_range);
                  double min_rule = min(var_split_range);

                  for(int cut = 0; cut < data.xcut.n_rows; cut++ ){
                       if((data.xcut(cut,g_node->var_split)>min_rule) & (data.xcut(cut,g_node->var_split)<max_rule)){
                            valid_cutpoint.push_back(data.xcut(cut,g_node->var_split));
                       }
                  }

             } else {

                  // Getting the maximum and the minimum;
                  for(int i = 0; i < g_node->n_leaf; i++){
                       var_split_range.push_back(data.x_train(g_node->train_index[i],g_node->var_split));
                  }

                  arma::vec var_split_range_arma = Rcpp::as<arma::vec>(var_split_range); // Conversion
                  arma::uvec conv_train_index = arma::conv_to<arma::uvec>::from(g_node->train_index);

                  arma::vec leaf_residuals = curr_res.elem(conv_train_index);

                  recoded_ranks = replaceWithUniqueRankMatrix(var_split_range_arma,leaf_residuals);
                  valid_cutpoint = Rcpp::wrap(recoded_ranks.col(1));
             }

             if(valid_cutpoint.size()==0){
                  p_try++;
                  if(data.sv_bool){
                       if(p_try>=sum_vars){
                            no_valid_node = true;
                       }
                  } else {
                       if(p_try>=data.x_train.n_cols){
                            no_valid_node = true;
                       };
                  }

             } else {
                  break; // Go out from the while
             }
        }

        if(no_valid_node){
             // Returning to the old values
             g_node->var_split = old_var_split;
             g_node->var_split_rule = old_var_split_rule;

             g_node->deletingLeaves();
             return;
        }


        // Selecting a rule (here I'm actually selecting the var split rule);
        g_node->var_split_rule = valid_cutpoint[arma::randi(arma::distr_param(0,valid_cutpoint.size()-1))];
        // cout << "The current var split rule is: " << g_node->var_split_rule << endl;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;

        arma::vec selected_predictor_train = data.x_train.col(g_node->var_split);
        arma::vec selected_predictor_test = data.x_test.col(g_node->var_split);

        if(data.categorical_indicators(g_node->var_split)!=0){

             selected_predictor_train = replaceWithMatrixMapping(selected_predictor_train,recoded_ranks);
             selected_predictor_test = replaceWithMatrixMapping(selected_predictor_test,recoded_ranks);

        }

        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
             if(g_node -> train_index[i] == -1 ){
                  g_node->left->n_leaf = train_left_counter;
                  g_node->right->n_leaf = train_right_counter;
                  break;
             }
             if(selected_predictor_train(g_node->train_index[i])<=g_node->var_split_rule){
                  g_node->left->train_index[train_left_counter] = g_node->train_index[i];
                  train_left_counter++;
             } else {
                  g_node->right->train_index[train_right_counter] = g_node->train_index[i];
                  train_right_counter++;
             }

        }



        // Updating the left and right nodes for the
        for(int i = 0;i<data.x_test.n_rows; i++){
             if(g_node -> test_index[i] == -1){
                  g_node->left->n_leaf_test = test_left_counter;
                  g_node->right->n_leaf_test = test_right_counter;
                  break;
             }
             if(selected_predictor_test(g_node->test_index[i])<=g_node->var_split_rule){
                  g_node->left->test_index[test_left_counter] = g_node->test_index[i];
                  test_left_counter++;
             } else {
                  g_node->right->test_index[test_right_counter] = g_node->test_index[i];
                  test_right_counter++;
             }
        }


        // If is a root node
        if(g_node->isRoot){
                g_node->left->n_leaf = train_left_counter;
                g_node->right->n_leaf = train_right_counter;
                g_node->left->n_leaf_test = test_left_counter;
                g_node->right->n_leaf_test = test_right_counter;
        }

        // Avoiding nodes lower than the node_min
        if((g_node->left->n_leaf<data.node_min_size) || (g_node->right->n_leaf<data.node_min_size) ){

                // cout << " NODES" << endl;
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;


                g_node->deletingLeaves();
                return;
        }




        // Updating the loglikelihood for those terminal nodes
        // cout << "Calculating likelihood of the new node on left" << endl;
        // cout << " ACCEPTED" << endl;
        g_node->left->updateResiduals(data,curr_res,curr_u);
        g_node->right->updateResiduals(data,curr_res,curr_u);


        // Calculating the likelihood on the new node on the left
        g_node->left->nodeLogLike(data);
        // cout << "Calculating likelihood of the new node on right" << endl;
        g_node->right->nodeLogLike(data);
        // cout << "NodeLogLike ok again" << endl;


        // Calculating the prior term for the grow
        double tree_prior = log(data.alpha*pow((1+g_node->depth),-data.beta)) +
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) + // Prior of left node being terminal
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) - // Prior of the right noide being terminal
                log(1-data.alpha*pow((1+g_node->depth),-data.beta)); // Old current node being terminal

        // Getting the transition probability
        double log_transition_prob = log((0.3)/(nog_nodes.size()+1)) - log(0.3/t_nodes.size()); // 0.3 and 0.3 are the prob of Prune and Grow, respectively

        // Calculating the loglikelihood for the new branches
        double new_tree_log_like = - g_node->log_likelihood + g_node->left->log_likelihood + g_node->right->log_likelihood ;

        // Calculating the acceptance ratio
        double acceptance = exp(new_tree_log_like  + log_transition_prob + tree_prior);


        // Keeping the new tree or not
        if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){
                // Do nothing just keep the new tree
                data.move_acceptance(0)++;
        } else {
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;

                g_node->deletingLeaves();
        }

        return;

}


// Grow a tree for a given rule
void grow_uni(Node* tree, modelParam &data, arma::vec &curr_res, int y_j_){

     // Getting the number of terminal nodes
     std::vector<Node*> t_nodes = leaves(tree) ;
     std::vector<Node*> nog_nodes = nogs(tree);

     // Selecting one node to be sampled
     Node* g_node = sample_node(t_nodes);

     // Store all old quantities that will be used or not
     int old_var_split = g_node->var_split;
     double old_var_split_rule = g_node->var_split_rule;

     // // Calculating the whole likelihood fo the tree
     for(int i = 0; i < t_nodes.size(); i++){
          // cout << "Error gpNodeLogLike" << endl;
          t_nodes[i]->updateResiduals_uni(data, curr_res); //
     }

     // Calculating the likelihood of the current grown node
     g_node->nodeLogLike(data);
     // cout << "LogLike Node ok Grow" << endl;

     // Adding the leaves
     g_node->addingLeaves(data);

     bool no_valid_node = false;
     int p_try = 0;
     // arma::cout << "variable (j): " << y_j_ << endl;
     int sum_vars = arma::sum(data.sv_matrix.row(y_j_));
     arma::vec split_candidates;

     // Trying to find a cutpoint
     if(data.sv_bool){
          arma::vec split_candidates_aux(sum_vars);
          int index_aux_ = 0;
          for(int ii_ = 0; ii_ < data.sv_matrix.n_cols; ii_++) {
               if(data.sv_matrix(y_j_,ii_)==1){
                    split_candidates_aux(index_aux_) = ii_;
                    index_aux_++;
               }
          }
          split_candidates = arma::shuffle(split_candidates_aux);

     } else {
          arma::vec split_candidates_aux = arma::shuffle(arma::regspace(0,1,data.x_train.n_cols-1));
          split_candidates = split_candidates_aux;
     }

     Rcpp::NumericVector valid_cutpoint;
     arma::mat recoded_ranks;

     while(!no_valid_node){
          g_node->var_split = split_candidates(p_try);

          Rcpp::NumericVector var_split_range;


          if(data.categorical_indicators(g_node->var_split)==0) {
               // Getting the maximum and the minimum;
               for(int i = 0; i < g_node->n_leaf; i++){
                    var_split_range.push_back(data.x_train(g_node->train_index[i],g_node->var_split));
               }

               // Getting the minimum and the maximum;
               double max_rule = max(var_split_range);
               double min_rule = min(var_split_range);

               for(int cut = 0; cut < data.xcut.n_rows; cut++ ){
                    if((data.xcut(cut,g_node->var_split)>min_rule) & (data.xcut(cut,g_node->var_split)<max_rule)){
                         valid_cutpoint.push_back(data.xcut(cut,g_node->var_split));
                    }
               }

          } else {

               // Getting the maximum and the minimum;
               for(int i = 0; i < g_node->n_leaf; i++){
                    var_split_range.push_back(data.x_train(g_node->train_index[i],g_node->var_split));
               }

               arma::vec var_split_range_arma = Rcpp::as<arma::vec>(var_split_range); // Conversion
               arma::uvec conv_train_index = arma::conv_to<arma::uvec>::from(g_node->train_index);

               arma::vec leaf_residuals = curr_res.elem(conv_train_index);

               recoded_ranks = replaceWithUniqueRankMatrix(var_split_range_arma,leaf_residuals);
               valid_cutpoint = Rcpp::wrap(recoded_ranks.col(1));
          }

          if(valid_cutpoint.size()==0){
               p_try++;
               if(data.sv_bool){
                    if(p_try>=sum_vars){
                         no_valid_node = true;
                    }
               } else {
                    if(p_try>=data.x_train.n_cols){
                         no_valid_node = true;
                    };
               }

          } else {
               break; // Go out from the while
          }
     }

     if(no_valid_node){
          // Returning to the old values
          g_node->var_split = old_var_split;
          g_node->var_split_rule = old_var_split_rule;

          g_node->deletingLeaves();
          return;
     }

     // Selecting a rule (here I'm actually selecting the var split rule);
     g_node->var_split_rule = valid_cutpoint[arma::randi(arma::distr_param(0,valid_cutpoint.size()-1))];
     // cout << "The current var split rule is: " << g_node->var_split_rule << endl;

     // Create an aux for the left and right index
     int train_left_counter = 0;
     int train_right_counter = 0;

     int test_left_counter = 0;
     int test_right_counter = 0;

     arma::vec selected_predictor_train = data.x_train.col(g_node->var_split);
     arma::vec selected_predictor_test = data.x_test.col(g_node->var_split);

     if(data.categorical_indicators(g_node->var_split)!=0){

          selected_predictor_train = replaceWithMatrixMapping(selected_predictor_train,recoded_ranks);
          selected_predictor_test = replaceWithMatrixMapping(selected_predictor_test,recoded_ranks);

     }

     // Updating the left and the right nodes
     for(int i = 0;i<data.x_train.n_rows;i++){
          if(g_node -> train_index[i] == -1 ){
               g_node->left->n_leaf = train_left_counter;
               g_node->right->n_leaf = train_right_counter;
               break;
          }
          if(selected_predictor_train(g_node->train_index[i])<=g_node->var_split_rule){
               g_node->left->train_index[train_left_counter] = g_node->train_index[i];
               train_left_counter++;
          } else {
               g_node->right->train_index[train_right_counter] = g_node->train_index[i];
               train_right_counter++;
          }

     }


     // Updating the left and right nodes for the
     for(int i = 0;i<data.x_test.n_rows; i++){
          if(g_node -> test_index[i] == -1){
               g_node->left->n_leaf_test = test_left_counter;
               g_node->right->n_leaf_test = test_right_counter;
               break;
          }
          if(selected_predictor_test(g_node->test_index[i])<=g_node->var_split_rule){
               g_node->left->test_index[test_left_counter] = g_node->test_index[i];
               test_left_counter++;
          } else {
               g_node->right->test_index[test_right_counter] = g_node->test_index[i];
               test_right_counter++;
          }
     }


     // If is a root node
     if(g_node->isRoot){
          g_node->left->n_leaf = train_left_counter;
          g_node->right->n_leaf = train_right_counter;
          g_node->left->n_leaf_test = test_left_counter;
          g_node->right->n_leaf_test = test_right_counter;
     }

     // Avoiding nodes lower than the node_min
     if((g_node->left->n_leaf<data.node_min_size) || (g_node->right->n_leaf<data.node_min_size) ){

          // cout << " NODES" << endl;
          // Returning to the old values
          g_node->var_split = old_var_split;
          g_node->var_split_rule = old_var_split_rule;

          g_node->deletingLeaves();
          return;
     }




     // Updating the loglikelihood for those terminal nodes
     // cout << "Calculating likelihood of the new node on left" << endl;
     // cout << " ACCEPTED" << endl;
     g_node->left->updateResiduals_uni(data,curr_res);
     g_node->right->updateResiduals_uni(data,curr_res);


     // Calculating the likelihood on the new node on the left
     g_node->left->nodeLogLike(data);
     // cout << "Calculating likelihood of the new node on right" << endl;
     g_node->right->nodeLogLike(data);
     // cout << "NodeLogLike ok again" << endl;


     // Calculating the prior term for the grow
     double tree_prior = log(data.alpha*pow((1+g_node->depth),-data.beta)) +
          log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) + // Prior of left node being terminal
          log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) - // Prior of the right noide being terminal
          log(1-data.alpha*pow((1+g_node->depth),-data.beta)); // Old current node being terminal

     // Getting the transition probability
     double log_transition_prob = log((0.3)/(nog_nodes.size()+1)) - log(0.3/t_nodes.size()); // 0.3 and 0.3 are the prob of Prune and Grow, respectively

     // Calculating the loglikelihood for the new branches
     double new_tree_log_like = - g_node->log_likelihood + g_node->left->log_likelihood + g_node->right->log_likelihood ;

     // Calculating the acceptance ratio
     double acceptance = exp(new_tree_log_like  + log_transition_prob + tree_prior);


     // Keeping the new tree or not
     if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){
          // Do nothing just keep the new tree
          data.move_acceptance(0)++;
     } else {
          // Returning to the old values
          g_node->var_split = old_var_split;
          g_node->var_split_rule = old_var_split_rule;

          g_node->deletingLeaves();
     }

     return;

}


// Pruning a tree
void prune(Node* tree, modelParam&data, arma::vec &curr_res, arma::vec &curr_u){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);

        // Can't prune a root
        if(t_nodes.size()==1){
                // cout << "Nodes size " << t_nodes.size() <<endl;
                t_nodes[0]->updateResiduals(data,curr_res,curr_u);
                t_nodes[0]->nodeLogLike(data);
                return;
        }

        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* p_node = sample_node(nog_nodes);


        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                t_nodes[i]->updateResiduals(data, curr_res,curr_u);
        }

        // cout << "Error C1" << endl;
        // Updating the loglikelihood of the selected pruned node
        p_node->updateResiduals(data,curr_res,curr_u);
        p_node->nodeLogLike(data);
        p_node->left->nodeLogLike(data);
        p_node->right->nodeLogLike(data);

        // Getting the loglikelihood of the new tree
        double new_tree_log_like =  p_node->log_likelihood - (p_node->left->log_likelihood + p_node->right->log_likelihood);

        // Calculating the transition loglikelihood
        double transition_loglike = log((0.3)/(t_nodes.size())) - log((0.3)/(nog_nodes.size()));

        // Calculating the prior term for the grow
        double tree_prior = log(1-data.alpha*pow((1+p_node->depth),-data.beta))-
                log(data.alpha*pow((1+p_node->depth),-data.beta)) -
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta)) - // Prior of left node being terminal
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta));  // Prior of the right noide being terminal
                 // Old current node being terminal


        // Calculating the acceptance
        double acceptance = exp(new_tree_log_like  + transition_loglike + tree_prior);

        if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
                p_node->deletingLeaves();
                data.move_acceptance(2)++;
        } else {
                // p_node->left->gpNodeLogLike(data, curr_res);
                // p_node->right->gpNodeLogLike(data, curr_res);
        }

        return;
}

// Pruning a tree
void prune_uni(Node* tree, modelParam&data, arma::vec &curr_res){


     // Getting the number of terminal nodes
     std::vector<Node*> t_nodes = leaves(tree);

     // Can't prune a root
     if(t_nodes.size()==1){
          // cout << "Nodes size " << t_nodes.size() <<endl;
          t_nodes[0]->updateResiduals_uni(data,curr_res);
          t_nodes[0]->nodeLogLike(data);
          return;
     }

     std::vector<Node*> nog_nodes = nogs(tree);

     // Selecting one node to be sampled
     Node* p_node = sample_node(nog_nodes);


     // Calculating the whole likelihood fo the tree
     for(int i = 0; i < t_nodes.size(); i++){
          t_nodes[i]->updateResiduals_uni(data, curr_res);
     }

     // cout << "Error C1" << endl;
     // Updating the loglikelihood of the selected pruned node
     p_node->updateResiduals_uni(data,curr_res);
     p_node->nodeLogLike(data);
     p_node->left->nodeLogLike(data);
     p_node->right->nodeLogLike(data);

     // Getting the loglikelihood of the new tree
     double new_tree_log_like =  p_node->log_likelihood - (p_node->left->log_likelihood + p_node->right->log_likelihood);

     // Calculating the transition loglikelihood
     double transition_loglike = log((0.3)/(t_nodes.size())) - log((0.3)/(nog_nodes.size()));

     // Calculating the prior term for the grow
     double tree_prior = log(1-data.alpha*pow((1+p_node->depth),-data.beta))-
          log(data.alpha*pow((1+p_node->depth),-data.beta)) -
          log(1-data.alpha*pow((1+p_node->depth+1),-data.beta)) - // Prior of left node being terminal
          log(1-data.alpha*pow((1+p_node->depth+1),-data.beta));  // Prior of the right noide being terminal
     // Old current node being terminal


     // Calculating the acceptance
     double acceptance = exp(new_tree_log_like  + transition_loglike + tree_prior);

     if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
          p_node->deletingLeaves();
          data.move_acceptance(2)++;
     } else {
          // p_node->left->gpNodeLogLike(data, curr_res);
          // p_node->right->gpNodeLogLike(data, curr_res);
     }

     return;
}


// // Creating the change verb
void change(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec &curr_u, int y_j_){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* c_node = sample_node(nog_nodes);


        if(c_node->isRoot){
                // cout << " THAT NEVER HAPPENS" << endl;
               c_node-> n_leaf = data.x_train.n_rows;
               c_node-> n_leaf_test = data.x_test.n_rows;
        }

        // cout << " Change error on terminal nodes" << endl;
        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Loglike error " << ed
                t_nodes[i]->updateResiduals(data, curr_res,curr_u);
        }

        // Calculating the loglikelihood of the old nodes
        c_node->left->nodeLogLike(data);
        c_node->right->nodeLogLike(data);


        // Storing all the old loglikelihood from left
        double old_left_log_like = c_node->left->log_likelihood;
        double old_left_r_sum = c_node->left->r_sum;
        double old_left_u_sum = c_node->left->u_sum;
        double old_left_Gamma_j = c_node->left->Gamma_j;
        double old_left_S_j = c_node->left->S_j;

        arma::vec old_left_train_index = c_node->left->train_index;
        c_node->left->train_index.fill(-1); // Returning to the original
        int old_left_n_leaf = c_node->left->n_leaf;


        // Storing all of the old loglikelihood from right;
        double old_right_log_like = c_node->right->log_likelihood;
        double old_right_r_sum = c_node->right->r_sum;
        double old_right_u_sum = c_node->right->u_sum;
        double old_right_Gamma_j = c_node->right->Gamma_j;
        double old_right_S_j = c_node->right->S_j;


        arma::vec old_right_train_index = c_node->right->train_index;
        c_node->right->train_index.fill(-1);
        int old_right_n_leaf = c_node->right->n_leaf;



        // Storing test observations
        arma::vec old_left_test_index = c_node->left->test_index;
        arma::vec old_right_test_index = c_node->right->test_index;
        c_node->left->test_index.fill(-1);
        c_node->right->test_index.fill(-1);

        int old_left_n_leaf_test = c_node->left->n_leaf_test;
        int old_right_n_leaf_test = c_node->right->n_leaf_test;


        // Storing the old ones
        int old_var_split = c_node->var_split;
        double old_var_split_rule = c_node->var_split_rule;
        double old_lower = c_node->lower;
        double old_upper = c_node->upper;

        // Choosing only valid cutpoints;
        bool no_valid_node = false;
        int p_try = 0;

        int sum_vars = arma::sum(data.sv_matrix.row(y_j_));
        arma::vec split_candidates;

        // Trying to find a cutpoint
        if(data.sv_bool){
                arma::vec split_candidates_aux(sum_vars);
                int index_aux_ = 0;
                for(int ii_ = 0; ii_ < data.sv_matrix.n_cols; ii_++) {
                        if(data.sv_matrix(y_j_,ii_)==1){
                                split_candidates_aux(index_aux_) = ii_;
                                index_aux_++;
                        }
                }
                split_candidates = arma::shuffle(split_candidates_aux);

        } else {
                arma::vec split_candidates_aux = arma::shuffle(arma::regspace(0,1,data.x_train.n_cols-1));
                split_candidates = split_candidates_aux;
        }

        // cout << " Split candidates: " << split_candidates << endl;
        // Trying to find a cutpoint
        Rcpp::NumericVector valid_cutpoint;
        arma::mat recoded_ranks;

        while(!no_valid_node){
             c_node->var_split = split_candidates(p_try);

             Rcpp::NumericVector var_split_range;


             if(data.categorical_indicators(c_node->var_split)==0) {
                  // Getting the maximum and the minimum;
                  for(int i = 0; i < c_node->n_leaf; i++){
                       var_split_range.push_back(data.x_train(c_node->train_index[i],c_node->var_split));
                  }

                  // Getting the minimum and the maximum;
                  double max_rule = max(var_split_range);
                  double min_rule = min(var_split_range);

                  for(int cut = 0; cut < data.xcut.n_rows; cut++ ){
                       if((data.xcut(cut,c_node->var_split)>min_rule) & (data.xcut(cut,c_node->var_split)<max_rule)){
                            valid_cutpoint.push_back(data.xcut(cut,c_node->var_split));
                       }
                  }

             } else {

                  // Getting the maximum and the minimum;
                  for(int i = 0; i < c_node->n_leaf; i++){
                       var_split_range.push_back(data.x_train(c_node->train_index[i],c_node->var_split));
                  }

                  arma::vec var_split_range_arma = Rcpp::as<arma::vec>(var_split_range); // Conversion
                  arma::uvec conv_train_index = arma::conv_to<arma::uvec>::from(c_node->train_index);

                  arma::vec leaf_residuals = curr_res.elem(conv_train_index);

                  recoded_ranks = replaceWithUniqueRankMatrix(var_split_range_arma,leaf_residuals);
                  valid_cutpoint = Rcpp::wrap(recoded_ranks.col(1));
             }

             if(valid_cutpoint.size()==0){
                  p_try++;
                  if(data.sv_bool){
                       if(p_try>=sum_vars){
                            no_valid_node = true;
                       }
                  } else {
                       if(p_try>=data.x_train.n_cols){
                            no_valid_node = true;
                       };
                  }

             } else {
                  break; // Go out from the while
             }
        }


        if(no_valid_node){
                // Returning to the old values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                return;
        }

        // Selecting a rule (here I'm actually selecting the var split rule);
        c_node->var_split_rule = valid_cutpoint[arma::randi(arma::distr_param(0,valid_cutpoint.size()-1))];

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;

        arma::vec selected_predictor_train = data.x_train.col(c_node->var_split);
        arma::vec selected_predictor_test = data.x_test.col(c_node->var_split);

        if(data.categorical_indicators(c_node->var_split)!=0){

             selected_predictor_train = replaceWithMatrixMapping(selected_predictor_train,recoded_ranks);
             selected_predictor_test = replaceWithMatrixMapping(selected_predictor_test,recoded_ranks);

        }

        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
             // cout << " Train indexeses " << c_node -> train_index[i] << endl ;
             if(c_node -> train_index[i] == -1){
                  c_node->left->n_leaf = train_left_counter;
                  c_node->right->n_leaf = train_right_counter;
                  break;
             }
             // cout << " Current train index " << c_node->train_index[i] << endl;

             if(selected_predictor_train(c_node->train_index[i])<=c_node->var_split_rule){
                  c_node->left->train_index[train_left_counter] = c_node->train_index[i];
                  train_left_counter++;
             } else {
                  c_node->right->train_index[train_right_counter] = c_node->train_index[i];
                  train_right_counter++;
             }
        }


        // Updating the left and the right nodes
        for(int i = 0;i<data.x_test.n_rows;i++){

             if(c_node -> test_index[i] == -1){
                  c_node->left->n_leaf_test = test_left_counter;
                  c_node->right->n_leaf_test = test_right_counter;
                  break;
             }

             if(selected_predictor_test(c_node->test_index[i])<=c_node->var_split_rule){
                  c_node->left->test_index[test_left_counter] = c_node->test_index[i];
                  test_left_counter++;
             } else {
                  c_node->right->test_index[test_right_counter] = c_node->test_index[i];
                  test_right_counter++;
             }
        }


        // If is a root node
        if(c_node->isRoot){
                c_node->left->n_leaf = train_left_counter;
                c_node->right->n_leaf = train_right_counter;
                c_node->left->n_leaf_test = test_left_counter;
                c_node->right->n_leaf_test = test_right_counter;
        }


        if((c_node->left->n_leaf<data.node_min_size) || (c_node->right->n_leaf)<data.node_min_size){

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->r_sum = old_left_r_sum;
                c_node->left->u_sum = old_left_u_sum;
                c_node->left->Gamma_j = old_left_Gamma_j ;
                c_node->left->S_j = old_left_S_j;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->r_sum = old_right_r_sum;
                c_node->right->u_sum = old_right_u_sum;
                c_node->right->Gamma_j = old_right_Gamma_j ;
                c_node->right->S_j = old_right_S_j;

                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

                return;
        }

        // Updating the new left and right loglikelihoods
        c_node->left->updateResiduals(data,curr_res,curr_u);
        c_node->right->updateResiduals(data,curr_res,curr_u);
        c_node->left->nodeLogLike(data);
        c_node->right->nodeLogLike(data);

        // Calculating the acceptance
        double new_tree_log_like =  - old_left_log_like - old_right_log_like + c_node->left->log_likelihood + c_node->right->log_likelihood;

        double acceptance = exp(new_tree_log_like);

        if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
                // Keep all the trees
                data.move_acceptance(2)++;
        } else {

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->r_sum = old_left_r_sum;
                c_node->left->u_sum = old_left_u_sum;
                c_node->left->Gamma_j = old_left_Gamma_j ;
                c_node->left->S_j = old_left_S_j;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->r_sum = old_right_r_sum;
                c_node->right->u_sum = old_right_u_sum;
                c_node->right->Gamma_j = old_right_Gamma_j ;
                c_node->right->S_j = old_right_S_j;

                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

        }

        return;
}

// Creating the change verb -- univariate version
void change_uni(Node* tree, modelParam &data, arma::vec &curr_res, int y_j_){


     // Getting the number of terminal nodes
     std::vector<Node*> t_nodes = leaves(tree) ;
     std::vector<Node*> nog_nodes = nogs(tree);

     // Selecting one node to be sampled
     Node* c_node = sample_node(nog_nodes);


     if(c_node->isRoot){
          // cout << " THAT NEVER HAPPENS" << endl;
          c_node-> n_leaf = data.x_train.n_rows;
          c_node-> n_leaf_test = data.x_test.n_rows;
     }

     // cout << " Change error on terminal nodes" << endl;
     // Calculating the whole likelihood fo the tree
     for(int i = 0; i < t_nodes.size(); i++){
          // cout << "Loglike error " << ed
          t_nodes[i]->updateResiduals_uni(data, curr_res);
     }

     // Calculating the loglikelihood of the old nodes
     c_node->left->nodeLogLike(data);
     c_node->right->nodeLogLike(data);


     // Storing all the old loglikelihood from left
     double old_left_log_like = c_node->left->log_likelihood;
     double old_left_r_sum = c_node->left->r_sum;
     double old_left_Gamma_j = c_node->left->Gamma_j;
     double old_left_S_j = c_node->left->S_j;

     arma::vec old_left_train_index = c_node->left->train_index;
     c_node->left->train_index.fill(-1); // Returning to the original
     int old_left_n_leaf = c_node->left->n_leaf;


     // Storing all of the old loglikelihood from right;
     double old_right_log_like = c_node->right->log_likelihood;
     double old_right_r_sum = c_node->right->r_sum;
     double old_right_Gamma_j = c_node->right->Gamma_j;
     double old_right_S_j = c_node->right->S_j;


     arma::vec old_right_train_index = c_node->right->train_index;
     c_node->right->train_index.fill(-1);
     int old_right_n_leaf = c_node->right->n_leaf;



     // Storing test observations
     arma::vec old_left_test_index = c_node->left->test_index;
     arma::vec old_right_test_index = c_node->right->test_index;
     c_node->left->test_index.fill(-1);
     c_node->right->test_index.fill(-1);

     int old_left_n_leaf_test = c_node->left->n_leaf_test;
     int old_right_n_leaf_test = c_node->right->n_leaf_test;


     // Storing the old ones
     int old_var_split = c_node->var_split;
     double old_var_split_rule = c_node->var_split_rule;
     double old_lower = c_node->lower;
     double old_upper = c_node->upper;

     // Choosing only valid cutpoints;
     bool no_valid_node = false;
     int p_try = 0;

     int sum_vars = arma::sum(data.sv_matrix.row(y_j_));
     arma::vec split_candidates;

     // Trying to find a cutpoint
     if(data.sv_bool){
          arma::vec split_candidates_aux(sum_vars);
          int index_aux_ = 0;
          for(int ii_ = 0; ii_ < data.sv_matrix.n_cols; ii_++) {
               if(data.sv_matrix(y_j_,ii_)==1){
                    split_candidates_aux(index_aux_) = ii_;
                    index_aux_++;
               }
          }
          split_candidates = arma::shuffle(split_candidates_aux);

     } else {
          arma::vec split_candidates_aux = arma::shuffle(arma::regspace(0,1,data.x_train.n_cols-1));
          split_candidates = split_candidates_aux;
     }

     // cout << " Split candidates: " << split_candidates << endl;
     // Trying to find a cutpoint
     Rcpp::NumericVector valid_cutpoint;
     arma::mat recoded_ranks;

     while(!no_valid_node){
          c_node->var_split = split_candidates(p_try);

          Rcpp::NumericVector var_split_range;


          if(data.categorical_indicators(c_node->var_split)==0) {
               // Getting the maximum and the minimum;
               for(int i = 0; i < c_node->n_leaf; i++){
                    var_split_range.push_back(data.x_train(c_node->train_index[i],c_node->var_split));
               }

               // Getting the minimum and the maximum;
               double max_rule = max(var_split_range);
               double min_rule = min(var_split_range);

               for(int cut = 0; cut < data.xcut.n_rows; cut++ ){
                    if((data.xcut(cut,c_node->var_split)>min_rule) & (data.xcut(cut,c_node->var_split)<max_rule)){
                         valid_cutpoint.push_back(data.xcut(cut,c_node->var_split));
                    }
               }

          } else {

               // Getting the maximum and the minimum;
               for(int i = 0; i < c_node->n_leaf; i++){
                    var_split_range.push_back(data.x_train(c_node->train_index[i],c_node->var_split));
               }

               arma::vec var_split_range_arma = Rcpp::as<arma::vec>(var_split_range); // Conversion
               arma::uvec conv_train_index = arma::conv_to<arma::uvec>::from(c_node->train_index);

               arma::vec leaf_residuals = curr_res.elem(conv_train_index);

               recoded_ranks = replaceWithUniqueRankMatrix(var_split_range_arma,leaf_residuals);
               valid_cutpoint = Rcpp::wrap(recoded_ranks.col(1));
          }

          if(valid_cutpoint.size()==0){
               p_try++;
               if(data.sv_bool){
                    if(p_try>=sum_vars){
                         no_valid_node = true;
                    }
               } else {
                    if(p_try>=data.x_train.n_cols){
                         no_valid_node = true;
                    };
               }

          } else {
               break; // Go out from the while
          }
     }

     if(no_valid_node){
          // Returning to the old values
          c_node->var_split = old_var_split;
          c_node->var_split_rule = old_var_split_rule;
          return;
     }

     // Selecting a rule (here I'm actually selecting the var split rule);
     c_node->var_split_rule = valid_cutpoint[arma::randi(arma::distr_param(0,valid_cutpoint.size()-1))];

     // Create an aux for the left and right index
     int train_left_counter = 0;
     int train_right_counter = 0;

     int test_left_counter = 0;
     int test_right_counter = 0;

     arma::vec selected_predictor_train = data.x_train.col(c_node->var_split);
     arma::vec selected_predictor_test = data.x_test.col(c_node->var_split);

     if(data.categorical_indicators(c_node->var_split)!=0){

          selected_predictor_train = replaceWithMatrixMapping(selected_predictor_train,recoded_ranks);
          selected_predictor_test = replaceWithMatrixMapping(selected_predictor_test,recoded_ranks);

     }

     // Updating the left and the right nodes
     for(int i = 0;i<data.x_train.n_rows;i++){
          // cout << " Train indexeses " << c_node -> train_index[i] << endl ;
          if(c_node -> train_index[i] == -1){
               c_node->left->n_leaf = train_left_counter;
               c_node->right->n_leaf = train_right_counter;
               break;
          }
          // cout << " Current train index " << c_node->train_index[i] << endl;

          if(selected_predictor_train(c_node->train_index[i])<=c_node->var_split_rule){
               c_node->left->train_index[train_left_counter] = c_node->train_index[i];
               train_left_counter++;
          } else {
               c_node->right->train_index[train_right_counter] = c_node->train_index[i];
               train_right_counter++;
          }
     }



     // Updating the left and the right nodes
     for(int i = 0;i<data.x_test.n_rows;i++){

          if(c_node -> test_index[i] == -1){
               c_node->left->n_leaf_test = test_left_counter;
               c_node->right->n_leaf_test = test_right_counter;
               break;
          }

          if(selected_predictor_test(c_node->test_index[i])<=c_node->var_split_rule){
               c_node->left->test_index[test_left_counter] = c_node->test_index[i];
               test_left_counter++;
          } else {
               c_node->right->test_index[test_right_counter] = c_node->test_index[i];
               test_right_counter++;
          }
     }

     // If is a root node
     if(c_node->isRoot){
          c_node->left->n_leaf = train_left_counter;
          c_node->right->n_leaf = train_right_counter;
          c_node->left->n_leaf_test = test_left_counter;
          c_node->right->n_leaf_test = test_right_counter;
     }


     if((c_node->left->n_leaf<data.node_min_size) || (c_node->right->n_leaf)<data.node_min_size){

          // Returning to the previous values
          c_node->var_split = old_var_split;
          c_node->var_split_rule = old_var_split_rule;
          c_node->lower = old_lower;
          c_node->upper = old_upper;

          // Returning to the old ones
          c_node->left->r_sum = old_left_r_sum;
          c_node->left->Gamma_j = old_left_Gamma_j ;
          c_node->left->S_j = old_left_S_j;

          c_node->left->n_leaf = old_left_n_leaf;
          c_node->left->n_leaf_test = old_left_n_leaf_test;
          c_node->left->log_likelihood = old_left_log_like;
          c_node->left->train_index = old_left_train_index;
          c_node->left->test_index = old_left_test_index;

          // Returning to the old ones
          c_node->right->r_sum = old_right_r_sum;
          c_node->right->Gamma_j = old_right_Gamma_j ;
          c_node->right->S_j = old_right_S_j;

          c_node->right->n_leaf = old_right_n_leaf;
          c_node->right->n_leaf_test = old_right_n_leaf_test;
          c_node->right->log_likelihood = old_right_log_like;
          c_node->right->train_index = old_right_train_index;
          c_node->right->test_index = old_right_test_index;

          return;
     }

     // Updating the new left and right loglikelihoods
     c_node->left->updateResiduals_uni(data,curr_res);
     c_node->right->updateResiduals_uni(data,curr_res);
     c_node->left->nodeLogLike(data);
     c_node->right->nodeLogLike(data);

     // Calculating the acceptance
     double new_tree_log_like =  - old_left_log_like - old_right_log_like + c_node->left->log_likelihood + c_node->right->log_likelihood;

     double acceptance = exp(new_tree_log_like);

     if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
          // Keep all the trees
          data.move_acceptance(2)++;
     } else {

          // Returning to the previous values
          c_node->var_split = old_var_split;
          c_node->var_split_rule = old_var_split_rule;
          c_node->lower = old_lower;
          c_node->upper = old_upper;

          // Returning to the old ones
          c_node->left->r_sum = old_left_r_sum;
          c_node->left->Gamma_j = old_left_Gamma_j ;
          c_node->left->S_j = old_left_S_j;

          c_node->left->n_leaf = old_left_n_leaf;
          c_node->left->n_leaf_test = old_left_n_leaf_test;
          c_node->left->log_likelihood = old_left_log_like;
          c_node->left->train_index = old_left_train_index;
          c_node->left->test_index = old_left_test_index;

          // Returning to the old ones
          c_node->right->r_sum = old_right_r_sum;
          c_node->right->Gamma_j = old_right_Gamma_j ;
          c_node->right->S_j = old_right_S_j;

          c_node->right->n_leaf = old_right_n_leaf;
          c_node->right->n_leaf_test = old_right_n_leaf_test;
          c_node->right->log_likelihood = old_right_log_like;
          c_node->right->train_index = old_right_train_index;
          c_node->right->test_index = old_right_test_index;

     }

     return;
}



// Calculating the Loglilelihood of a node
void Node::updateResiduals(modelParam& data,
                           arma::vec &curr_res,
                           arma::vec &curr_u){

        // Getting number of leaves in case of a root
        if(isRoot){
                // Updating the r_sum
                n_leaf = data.x_train.n_rows;
                n_leaf_test = data.x_test.n_rows;
        }


        // Case of an empty node
        if(train_index[0]==-1){
        // if(n_leaf < 100){
                n_leaf = 0;
                r_sum = 0;
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // If is smaller then the node size still need to update the quantities;
        // cout << "Node min size: " << data.node_min_size << endl;
        if(n_leaf < data.node_min_size){
                // log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        r_sum = 0.0;
        u_sum = 0.0;

        // Train elements
        for(int i = 0; i < n_leaf;i++){
                r_sum = r_sum + curr_res(train_index[i]);
                u_sum = u_sum + curr_u(train_index[i]);
        }

        sigma_mu_j_sq = data.sigma_mu_j*data.sigma_mu_j;
        Gamma_j  = n_leaf+data.v_j/sigma_mu_j_sq;
        S_j = r_sum-u_sum;

        return;

}

// Calculating the Loglilelihood of a node
void Node::updateResiduals_uni(modelParam& data,
                              arma::vec &curr_res){

     // Getting number of leaves in case of a root
     if(isRoot){
          // Updating the r_sum
          n_leaf = data.x_train.n_rows;
          n_leaf_test = data.x_test.n_rows;
     }


     // Case of an empty node
     if(train_index[0]==-1){
          // if(n_leaf < 100){
          n_leaf = 0;
          r_sum = 0;
          log_likelihood = -2000000; // Absurd value avoid this case
          // cout << "OOOPS something happened" << endl;
          return;
     }

     // If is smaller then the node size still need to update the quantities;
     // cout << "Node min size: " << data.node_min_size << endl;
     if(n_leaf < data.node_min_size){
          // log_likelihood = -2000000; // Absurd value avoid this case
          // cout << "OOOPS something happened" << endl;
          return;
     }

     r_sum = 0.0;
     u_sum = 0.0;
     // Train elements
     for(int i = 0; i < n_leaf;i++){
          r_sum = r_sum + curr_res(train_index[i]);
     }

     sigma_mu_j_sq = data.sigma_mu_j*data.sigma_mu_j;
     Gamma_j  = n_leaf+data.v_j/sigma_mu_j_sq;
     S_j = r_sum;

     return;

}

void Node::nodeLogLike(modelParam& data){
        // Getting the log-likelihood;
        // double sigma_mu_j_sq = data.sigma_mu_j*data.sigma_mu_j;

        log_likelihood = -0.5*log(2*arma::datum::pi*sigma_mu_j_sq)+0.5*log(data.v_j/Gamma_j) +0.5*(S_j*S_j)/(data.v_j*Gamma_j);
        return;
}


//Updating Mu and Predictions at the same time
void updateMuPredictions(Node* tree, modelParam &data,
                         arma::vec &current_prediction_train,
                         arma::vec &current_prediction_test){

        // Getting the terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);


        // Iterating over the terminal nodes and updating the beta values
        for(int i = 0; i < t_nodes.size();i++){

                // Updating mu
                t_nodes[i]->mu = R::rnorm((t_nodes[i]->S_j)/(t_nodes[i]->Gamma_j),sqrt(data.v_j/(t_nodes[i]->Gamma_j))) ;

                // Create a boolean to update the train
                bool update_train_ = true;
                bool update_test_ = true;
                int for_size_;

                // Setting the size of the for
                if(data.x_train.n_rows > data.x_test.n_rows){
                        for_size_ = data.x_train.n_rows;
                } else {
                        for_size_ = data.x_test.n_rows;
                }

                // Iterating over the training.rows
                for(int ii = 0; ii < for_size_ ; ii++){


                        // Avoiding memory crashing of checking ->train_index[ii]
                        if(ii >= data.x_train.n_rows){
                                update_train_ = false;
                        } else {
                                // Updating for the training samples
                                if(t_nodes[i]->train_index[ii] == -1){
                                        update_train_ = false;
                                }
                        }

                        // Avoiding memory crashing of checking ->test_index[ii]
                        if(ii >= data.x_test.n_rows){
                                update_test_ = false;
                        } else {
                                // Updating for the test samples
                                if(t_nodes[i]->test_index[ii] == -1){
                                        update_test_ = false;
                                }
                        }


                        if(update_train_){
                                current_prediction_train[t_nodes[i]->train_index[ii]] = t_nodes[i] -> mu;
                        }


                        // Getting out of the for if necessary
                        if(!(update_train_) & !(update_test_)){
                                break; // Get out from the for
                        }

                        if(t_nodes[i]->n_leaf_test == 0 ){
                                continue;
                        }

                        if(update_test_){
                                current_prediction_test[t_nodes[i]->test_index[ii]] = t_nodes[i] -> mu;
                        }


                }
        }
}

// UPDATING MU
void updateMu(Node* tree, modelParam &data, arma::vec &curr_r, arma::vec &curr_u){

        // Getting the terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);


        // Iterating over the terminal nodes and updating the beta values
        for(int i = 0; i < t_nodes.size();i++){
                // double s_j = 0;
                // double gamma_j = 0;

                // for(int k = 0; k < t_nodes[i]->train_index.size(); k++){
                //         s_j = curr
                // }

                t_nodes[i]->mu = R::rnorm((t_nodes[i]->S_j)/(t_nodes[i]->Gamma_j),sqrt(data.v_j/(t_nodes[i]->Gamma_j))) ;

        }
}

void updateSigma(arma::mat &y_mat_hat,
                 modelParam &data){

        arma::mat S(data.y_mat.n_cols,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat residuals_mat = y_mat_hat-data.y_mat;
        S = residuals_mat.t()*residuals_mat;

        // Updating sigma
        data.Sigma = arma::iwishrnd((data.S_0_wish+S),data.nu + data.y_mat.n_cols - 1 + data.y_mat.n_rows);

}




// Get the prediction
void getPredictions(Node* tree,
                    modelParam &data,
                    arma::vec& current_prediction_train,
                    arma::vec& current_prediction_test){

        // Getting the current prediction
        vector<Node*> t_nodes = leaves(tree);
        for(int i = 0; i<t_nodes.size();i++){

                // Skipping empty nodes
                if(t_nodes[i]->n_leaf==0){
                        Rcpp::Rcout << " THERE ARE EMPTY NODES" << endl;
                        continue;
                }


                // For the training samples
                for(int j = 0; j<data.x_train.n_rows; j++){

                        if((t_nodes[i]->train_index[j])==-1){
                                break;
                        }
                        current_prediction_train[t_nodes[i]->train_index[j]] = t_nodes[i]->mu;
                }

                if(t_nodes[i]->n_leaf_test == 0 ){
                        continue;
                }



                // Regarding the test samples
                for(int j = 0; j< data.x_test.n_rows;j++){

                        if(t_nodes[i]->test_index[j]==-1){
                                break;
                        }

                        current_prediction_test[t_nodes[i]->test_index[j]] = t_nodes[i]->mu;
                }

        }
}


void update_a_j(modelParam &data){

        double shape_j = 0.5*(data.y_mat.n_cols+data.nu);
        arma::mat Precision = arma::inv(data.Sigma);

        // Rcpp::Rcout << "a_j_vec is: "<< data.a_j_vec.size() << endl;
        // Rcpp::Rcout << "A_j_vec is: "<< data.a_j_vec.size() << endl;
        // Rcpp::Rcout << "S_0_vec is: "<< data.S_0_wish.size() << endl;

        // Calcularting shape and scale parameters
        for(int j = 0; j < data.y_mat.n_cols; j++){
                double scale_j = 1/(data.A_j_vec(j)*data.A_j_vec(j))+data.nu*Precision(j,j);
                double a_j_vec_double_aux = R::rgamma(shape_j,1/scale_j);
                data.a_j_vec(j) = 1/a_j_vec_double_aux;
                data.S_0_wish(j,j) = (2*data.nu)/data.a_j_vec(j);
                // Rcpp::Rcout << " Iteration j" << endl;
        }

        return;
}

// CPP-BART function
// [[Rcpp::export]]
Rcpp::List cppbart(arma::mat x_train,
          arma::mat y_mat,
          arma::mat x_test,
          arma::mat x_cut,
          int n_tree,
          int node_min_size,
          int n_mcmc,
          int n_burn,
          arma::mat Sigma_init,
          arma::vec mu_init,
          arma::vec sigma_mu,
          double alpha, double beta, double nu,
          arma::mat S_0_wish,
          arma::vec A_j_vec,
          bool update_Sigma,
          bool var_selection_bool,
          bool sv_bool,
          bool hier_prior_bool,
          arma::mat sv_matrix,
          arma::vec categorical_indicators){

        // Posterior counter
        int curr = 0;


        // cout << " Error on model.param" << endl;
        // Creating the structu object
        modelParam data(x_train,
                        y_mat,
                        x_test,
                        x_cut,
                        n_tree,
                        node_min_size,
                        alpha,
                        beta,
                        nu,
                        sigma_mu,
                        Sigma_init,
                        S_0_wish,
                        A_j_vec,
                        n_mcmc,
                        n_burn,
                        sv_bool,
                        sv_matrix,
                        categorical_indicators);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;

        // Rcpp::Rcout << "error here" << endl;

        // Defining those elements
        arma::cube y_train_hat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube y_test_hat_post(data.x_test.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_post,arma::fill::zeros);
        arma::cube all_Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_mcmc,arma::fill::zeros);

        // Rcpp::Rcout << "error here2" << endl;

        // =====================================
        // For the moment I will not store those
        // =====================================
        // arma::cube all_tree_post(y_mat.size(),n_tree,n_post,arma::fill::zeros);


        // Defining other variables
        // arma::vec partial_pred = arma::mat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::vec partial_residuals(data.x_train.n_rows,arma::fill::zeros);
        arma::cube tree_fits_store(data.x_train.n_rows,data.n_tree,data.y_mat.n_cols,arma::fill::zeros);
        arma::cube tree_fits_store_test(data.x_test.n_rows,data.n_tree,y_mat.n_cols,arma::fill::zeros);

        // Setting a vector to store the variables used in the split
        arma::field<arma::cube> all_j_tree_var(data.n_mcmc);
        for(int cube_dim = 0; cube_dim <all_j_tree_var.size(); cube_dim ++){
                all_j_tree_var(cube_dim) = arma::cube(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);
        }


        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;

        // Selecting the train
        Forest all_forest(data);

        // Creating variables to help define in which tree set we are;
        int curr_tree_counter;

        // Matrix that store all the predictions for all y
        arma::mat y_mat_hat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat y_mat_test_hat(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);


        for(int i = 0;i<data.n_mcmc;i++){

                // Initialising PB
                Rcpp::Rcout << "[";
                int k = 0;

                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        Rcpp::Rcout << "=";
                }

                for(; k < width;k++){
                        Rcpp::Rcout << " ";
                }

                Rcpp::Rcout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                Rcpp::Rcout.flush();


                // Getting zeros
                arma::mat prediction_train_sum(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
                arma::mat prediction_test_sum(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);


                arma::vec partial_u(data.x_train.n_rows);
                arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
                arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);

                // Aux for used vars
                arma::cube cube_used_vars(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);

                // Iterating over the d-dimension MATRICES of the response.
                for(int j = 0; j < data.y_mat.n_cols; j++){


                        // Storing all used vars
                        arma::mat mat_used_vars(data.n_tree, data.x_train.n_cols);

                        // Covariance objects
                        arma::mat Sigma_j_mj(1,(data.y_mat.n_cols-1),arma::fill::zeros);
                        arma::mat Sigma_mj_j((data.y_mat.n_cols-1),1,arma::fill::zeros);
                        arma::mat Sigma_mj_mj = data.Sigma;


                        double Sigma_j_j = data.Sigma(j,j);

                        int aux_j_counter = 0;


                        // Dropping the column with respect to "j"
                        Sigma_mj_mj.shed_row(j);
                        Sigma_mj_mj.shed_col(j);

                        for(int d = 0; d < data.y_mat.n_cols; d++){

                                // Rcpp::Rcout <<  "AUX_J: " << data.Sigma(j,d) << endl;
                                if(d!=j){
                                        Sigma_j_mj(0,aux_j_counter)  = data.Sigma(j,d);
                                        Sigma_mj_j(aux_j_counter,0) = data.Sigma(j,d);
                                        aux_j_counter = aux_j_counter + 1;
                                }

                        }


                        // ============================================
                        // This step does not iterate over the trees!!!
                        // ===========================================
                        y_mj = data.y_mat;
                        y_mj.shed_col(j);
                        y_hat_mj = y_mat_hat;
                        y_hat_mj.shed_col(j);

                        // Calculating the invertion that gonna be used for the U and V
                        arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);
                        arma::mat Sigma_calculation_aux = (Sigma_mj_j.t()*Sigma_mj_mj_inv); // This line calculate a quantitiy that is constant for the i_train so avoids multiple calculations;

                                // Calculating the current partial U
                                for(int i_train = 0; i_train < data.y_mat.n_rows;i_train++){
                                                partial_u(i_train) = arma::as_scalar(Sigma_calculation_aux*(y_mj.row(i_train)-y_hat_mj.row(i_train)).t()); // Old version

                                }

                                double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj*Sigma_mj_mj_inv*Sigma_mj_j);
                                data.v_j = v;




                        data.sigma_mu_j = data.sigma_mu(j);


                        // Updating the tree
                        for(int t = 0; t<data.n_tree;t++){

                                // Current tree counter
                                curr_tree_counter = t + j*data.n_tree;

                                // Creating the auxliar prediction vector
                                arma::vec y_j_hat(data.y_mat.n_rows,arma::fill::zeros);
                                arma::vec y_j_test_hat(data.x_test.n_rows,arma::fill::zeros);

                                // Updating the partial residuals
                                if(data.n_tree>1){
                                        partial_residuals = data.y_mat.col(j)-sum_exclude_col(tree_fits_store.slice(j),t);
                                } else {
                                        partial_residuals = data.y_mat.col(j);
                                }

                                // Iterating over all trees
                                verb = arma::randu(arma::distr_param(0.0,1.0));

                                if(all_forest.trees[curr_tree_counter]->isLeaf & all_forest.trees[curr_tree_counter]->isRoot){
                                        verb = 0.1;
                                }

                                // Selecting the verb
                                if(verb < 0.25){
                                        data.move_proposal(0)++;
                                        grow(all_forest.trees[curr_tree_counter],data,partial_residuals,partial_u,j);
                                } else if(verb>=0.25 & verb <0.5) {
                                        data.move_proposal(1)++;
                                        prune(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u);
                                } else {
                                        data.move_proposal(2)++;
                                        change(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u,j);
                                }


                                // Updating Mu and Prediction
                                updateMuPredictions(all_forest.trees[curr_tree_counter],data,y_j_hat,y_j_test_hat);

                                // Updating the tree
                                // cout << "Residuals error 2.0"<< endl;
                                tree_fits_store.slice(j).col(t) = y_j_hat;
                                // cout << "Residuals error 3.0"<< endl;
                                tree_fits_store_test.slice(j).col(t) = y_j_test_hat;
                                // cout << "Residuals error 4.0"<< endl;

                                // Aux for the used vars
                                if(var_selection_bool){
                                        arma::vec used_vars(data.x_train.n_cols);
                                        collect_split_vars(used_vars,all_forest.trees[curr_tree_counter]);
                                        mat_used_vars.row(t) = used_vars.t();
                                }
                                // arma::cout << "Used vars on tree: " << used_vars << endl;

                        } // End of iterations over "t"

                        // Summing over all trees
                        prediction_train_sum = sum(tree_fits_store.slice(j),1);
                        y_mat_hat.col(j) = prediction_train_sum;

                        prediction_test_sum = sum(tree_fits_store_test.slice(j),1);
                        y_mat_test_hat.col(j) = prediction_test_sum;

                        cube_used_vars.slice(j) = mat_used_vars;
                }// End of iterations over "j"


                // Storing cube of used vars
                all_j_tree_var(i) = cube_used_vars;

                // std::cout << "Error Tau: " << data.tau<< endl;
                if(update_Sigma){
                        // Updating or not a_j
                        if(hier_prior_bool){
                                update_a_j(data);
                        }
                        updateSigma(y_mat_hat, data);
                }

                all_Sigma_post.slice(i) = data.Sigma;

                // std::cout << " All good " << endl;
                if(i >= n_burn){
                        // Storing the predictions
                        y_train_hat_post.slice(curr) = y_mat_hat;
                        y_test_hat_post.slice(curr) = y_mat_test_hat;
                        Sigma_post.slice(curr) = data.Sigma;
                        curr++;
                }

                pb += 1;

        }
        // Initialising PB
        Rcpp::Rcout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                Rcpp::Rcout << "=";
        }

        for(; k < width;k++){
                Rcpp::Rcout << " ";
        }

        Rcpp::Rcout << "] " << std::setprecision(5) << 100 << "%\r";
        Rcpp::Rcout.flush();

        Rcpp::Rcout << std::endl;

        return Rcpp::List::create(y_train_hat_post, //[1]
                                  y_test_hat_post, //[2]
                                  Sigma_post, //[3]
                                  all_Sigma_post, // [4]
                                  data.move_proposal, // [5]
                                  data.move_acceptance, //[6]
                                  all_j_tree_var //[7]
                                );
}

// [[Rcpp::export]]
Rcpp::List cppbart_univariate(arma::mat x_train,
                   arma::mat y_mat,
                   arma::mat x_test,
                   arma::mat x_cut,
                   int n_tree,
                   int node_min_size,
                   int n_mcmc,
                   int n_burn,
                   arma::mat Sigma_init,
                   arma::vec mu_init,
                   arma::vec sigma_mu,
                   double alpha, double beta, double nu,
                   arma::mat S_0_wish,
                   arma::vec A_j_vec,
                   bool update_Sigma,
                   bool var_selection_bool,
                   bool sv_bool,
                   bool hier_prior_bool,
                   arma::mat sv_matrix,
                   arma::vec categorical_indicators){

     // Posterior counter
     int curr = 0;

     if(y_mat.n_cols!=1){
          Rcpp::stop("Do not call cppbart_univariate if there's than one column");
     }

     // cout << " Error on model.param" << endl;
     // Creating the structu object
     modelParam data(x_train,
                     y_mat,
                     x_test,
                     x_cut,
                     n_tree,
                     node_min_size,
                     alpha,
                     beta,
                     nu,
                     sigma_mu,
                     Sigma_init,
                     S_0_wish,
                     A_j_vec,
                     n_mcmc,
                     n_burn,
                     sv_bool,
                     sv_matrix,
                     categorical_indicators);

     // Getting the n_post
     int n_post = n_mcmc - n_burn;

     // Rcpp::Rcout << "error here" << endl;

     // Defining those elements
     arma::cube y_train_hat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
     arma::cube y_test_hat_post(data.x_test.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
     arma::cube Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_post,arma::fill::zeros);
     arma::cube all_Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_mcmc,arma::fill::zeros);

     // Rcpp::Rcout << "error here2" << endl;

     // =====================================
     // For the moment I will not store those
     // =====================================
     // arma::cube all_tree_post(y_mat.size(),n_tree,n_post,arma::fill::zeros);


     // Defining other variables
     // arma::vec partial_pred = arma::mat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
     arma::vec partial_residuals(data.x_train.n_rows,arma::fill::zeros);
     arma::cube tree_fits_store(data.x_train.n_rows,data.n_tree,data.y_mat.n_cols,arma::fill::zeros);
     arma::cube tree_fits_store_test(data.x_test.n_rows,data.n_tree,y_mat.n_cols,arma::fill::zeros);

     // Setting a vector to store the variables used in the split
     arma::field<arma::cube> all_j_tree_var(data.n_mcmc);
     for(int cube_dim = 0; cube_dim <all_j_tree_var.size(); cube_dim ++){
          all_j_tree_var(cube_dim) = arma::cube(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);
     }


     double verb;

     // Defining progress bars parameters
     const int width = 70;
     double pb = 0;

     // Selecting the train
     Forest all_forest(data);

     // Creating variables to help define in which tree set we are;
     int curr_tree_counter;

     // Matrix that store all the predictions for all y
     arma::mat y_mat_hat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
     arma::mat y_mat_test_hat(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);


     for(int i = 0;i<data.n_mcmc;i++){

          // Initialising PB
          Rcpp::Rcout << "[";
          int k = 0;

          // Evaluating progress bar
          for(;k<=pb*width/data.n_mcmc;k++){
               Rcpp::Rcout << "=";
          }

          for(; k < width;k++){
               Rcpp::Rcout << " ";
          }

          Rcpp::Rcout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
          Rcpp::Rcout.flush();


          // Getting zeros
          arma::mat prediction_train_sum(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
          arma::mat prediction_test_sum(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

          // Aux for used vars
          arma::cube cube_used_vars(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);

          // Iterating over the d-dimension MATRICES of the response.
          for(int j = 0; j < data.y_mat.n_cols; j++){

               //Rcpp::Rcout << "could enter the loop" << endl;
               // Storing all used vars
               arma::mat mat_used_vars(data.n_tree, data.x_train.n_cols);

               // Rcpp::Rcout << "could enter the loop" << endl;
               // Rcpp::Rcout << "DImensions of Sigma: " << data.Sigma.size() << endl;
               double Sigma_j_j = data.Sigma(j,j);

               double v = Sigma_j_j;
               data.v_j = v;

               data.sigma_mu_j = data.sigma_mu(j);


               // Updating the tree
               for(int t = 0; t<data.n_tree;t++){

                    // Current tree counter
                    curr_tree_counter = t + j*data.n_tree;

                    // Creating the auxliar prediction vector
                    arma::vec y_j_hat(data.y_mat.n_rows,arma::fill::zeros);
                    arma::vec y_j_test_hat(data.x_test.n_rows,arma::fill::zeros);

                    // Updating the partial residuals
                    // Rcpp::Rcout << "Error on partial residuals calculation " << endl;
                    if(data.n_tree>1){
                         partial_residuals = data.y_mat.col(j)-sum_exclude_col(tree_fits_store.slice(j),t);
                    } else {
                         partial_residuals = data.y_mat.col(j);
                    }

                    // Iterating over all trees
                    verb = arma::randu(arma::distr_param(0.0,1.0));

                    if(all_forest.trees[curr_tree_counter]->isLeaf & all_forest.trees[curr_tree_counter]->isRoot){
                         verb = 0.1;
                    }

                    // Rcpp::Rcout << "Problem before the verb" << endl;
                    // Selecting the verb
                    if(verb < 0.25){
                         data.move_proposal(0)++;
                         grow_uni(all_forest.trees[curr_tree_counter],data,partial_residuals,j);
                    } else if(verb>=0.25 & verb <0.5) {
                         data.move_proposal(1)++;
                         prune_uni(all_forest.trees[curr_tree_counter], data, partial_residuals);
                    } else {
                         data.move_proposal(2)++;
                         change_uni(all_forest.trees[curr_tree_counter], data, partial_residuals,j);
                    }
                    // Rcpp::Rcout << "Problem after the verb" << endl;



                    // Updating Mu and Prediction
                    updateMuPredictions(all_forest.trees[curr_tree_counter],data,y_j_hat,y_j_test_hat);

                    // Updating the tree
                    // cout << "Residuals error 2.0"<< endl;
                    tree_fits_store.slice(j).col(t) = y_j_hat;
                    // cout << "Residuals error 3.0"<< endl;
                    tree_fits_store_test.slice(j).col(t) = y_j_test_hat;
                    // cout << "Residuals error 4.0"<< endl;

                    // Aux for the used vars
                    if(var_selection_bool){
                         arma::vec used_vars(data.x_train.n_cols);
                         collect_split_vars(used_vars,all_forest.trees[curr_tree_counter]);
                         mat_used_vars.row(t) = used_vars.t();
                    }
                    // arma::cout << "Used vars on tree: " << used_vars << endl;

               } // End of iterations over "t"

               // Summing over all trees
               prediction_train_sum = sum(tree_fits_store.slice(j),1);
               y_mat_hat.col(j) = prediction_train_sum;

               prediction_test_sum = sum(tree_fits_store_test.slice(j),1);
               y_mat_test_hat.col(j) = prediction_test_sum;

               cube_used_vars.slice(j) = mat_used_vars;
          }// End of iterations over "j"


          // Storing cube of used vars
          all_j_tree_var(i) = cube_used_vars;

          // std::cout << "Error Tau: " << endl;

          if(update_Sigma){
               // Updating or not a_j
               if(hier_prior_bool){
                    update_a_j(data);
               }
               // std::cout << "Error SIGMA: " << endl;

               updateSigma(y_mat_hat, data);
          }

          all_Sigma_post.slice(i) = data.Sigma;

          // std::cout << " All good " << endl;
          if(i >= n_burn){
               // Storing the predictions
               y_train_hat_post.slice(curr) = y_mat_hat;
               y_test_hat_post.slice(curr) = y_mat_test_hat;
               Sigma_post.slice(curr) = data.Sigma;
               curr++;
          }

          pb += 1;

     }
     // Initialising PB
     Rcpp::Rcout << "[";
     int k = 0;
     // Evaluating progress bar
     for(;k<=pb*width/data.n_mcmc;k++){
          Rcpp::Rcout << "=";
     }

     for(; k < width;k++){
          Rcpp::Rcout << " ";
     }

     Rcpp::Rcout << "] " << std::setprecision(5) << 100 << "%\r";
     Rcpp::Rcout.flush();

     Rcpp::Rcout << std::endl;

     return Rcpp::List::create(y_train_hat_post, //[1]
                               y_test_hat_post, //[2]
                               Sigma_post, //[3]
                               all_Sigma_post, // [4]
                               data.move_proposal, // [5]
                               data.move_acceptance, //[6]
                               all_j_tree_var //[7]
     );
}


void update_y_mat_missing(modelParam & data,
                          arma::mat & y_mat_hat,
                          arma::mat & na_indicators,
                          int ii) {


        // Creating the copies dropping the ii column
        arma::mat y_mat_mj = data.y_mat;
        arma::mat y_hat_mj = y_mat_hat;
        arma::mat Sigma_mj_mj = data.Sigma;
        arma::mat Sigma_j_mj = data.Sigma;
        arma::mat Sigma_mj_j = data.Sigma;

        y_mat_mj.shed_col(ii);
        y_hat_mj.shed_col(ii);
        Sigma_mj_mj.shed_col(ii);
        Sigma_mj_mj.shed_row(ii);
        Sigma_j_mj.shed_col(ii);
        Sigma_mj_j.shed_row(ii);

        //Rcpp::Rcout << "Value for ii:" << ii << endl;
        arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);
        arma::mat scale_mean_aux = Sigma_j_mj.row(ii)*Sigma_mj_mj_inv;

        double variance_aux = data.Sigma(ii,ii) - arma::as_scalar(scale_mean_aux*Sigma_mj_j.col(ii));
        double mean_y_ii;

        for(int kk = 0; kk < na_indicators.n_rows;kk++){
                if(na_indicators(kk,ii)==1){
                        mean_y_ii = arma::as_scalar(y_mat_hat(kk,ii) + scale_mean_aux*(y_mat_mj.row(kk)-y_hat_mj.row(kk)));
                        // Rcpp::Rcout << " Printing the "
                        data.y_mat(kk,ii) = R::rnorm(mean_y_ii,sqrt(variance_aux));
                }
        }

        return;
}
// [[Rcpp::export]]
Rcpp::List cppbart_missing(arma::mat x_train,
                   arma::mat y_mat,
                   arma::vec n_missing,
                   arma::mat na_indicators,
                   arma::mat x_test,
                   arma::mat x_cut,
                   int n_tree,
                   int node_min_size,
                   int n_mcmc,
                   int n_burn,
                   arma::mat Sigma_init,
                   arma::vec mu_init,
                   arma::vec sigma_mu,
                   double alpha, double beta, double nu,
                   arma::mat S_0_wish,
                   arma::vec A_j_vec,
                   bool update_Sigma,
                   bool var_selection_bool,
                   bool sv_bool,
                   bool hier_prior_bool,
                   arma::mat sv_matrix,
                   arma::vec categorical_indicators){

        // Posterior counter
        int curr = 0;


        // cout << " Error on model.param" << endl;
        // Creating the structu object
        modelParam data(x_train,
                        y_mat,
                        x_test,
                        x_cut,
                        n_tree,
                        node_min_size,
                        alpha,
                        beta,
                        nu,
                        sigma_mu,
                        Sigma_init,
                        S_0_wish,
                        A_j_vec,
                        n_mcmc,
                        n_burn,
                        sv_bool,
                        sv_matrix,
                        categorical_indicators);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;

        // Rcpp::Rcout << "error here" << endl;

        // Defining those elements
        arma::cube y_train_hat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube y_test_hat_post(data.x_test.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube y_mat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post, arma::fill::zeros);
        arma::cube Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_post,arma::fill::zeros);
        arma::cube all_Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_mcmc,arma::fill::zeros);

        // Rcpp::Rcout << "error here2" << endl;

        // =====================================
        // For the moment I will not store those
        // =====================================
        // arma::cube all_tree_post(y_mat.size(),n_tree,n_post,arma::fill::zeros);


        // Defining other variables
        // arma::vec partial_pred = arma::mat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::vec partial_residuals(data.x_train.n_rows,arma::fill::zeros);
        arma::cube tree_fits_store(data.x_train.n_rows,data.n_tree,data.y_mat.n_cols,arma::fill::zeros);
        arma::cube tree_fits_store_test(data.x_test.n_rows,data.n_tree,y_mat.n_cols,arma::fill::zeros);

        // Setting a vector to store the variables used in the split
        arma::field<arma::cube> all_j_tree_var(data.n_mcmc);
        for(int cube_dim = 0; cube_dim <all_j_tree_var.size(); cube_dim ++){
                all_j_tree_var(cube_dim) = arma::cube(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);
        }


        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;

        // Selecting the train
        Forest all_forest(data);

        // Creating variables to help define in which tree set we are;
        int curr_tree_counter;

        // Matrix that store all the predictions for all y
        arma::mat y_mat_hat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat y_mat_test_hat(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);


        for(int i = 0;i<data.n_mcmc;i++){

                // Initialising PB
                Rcpp::Rcout << "[";
                int k = 0;

                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        Rcpp::Rcout << "=";
                }

                for(; k < width;k++){
                        Rcpp::Rcout << " ";
                }

                Rcpp::Rcout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                Rcpp::Rcout.flush();


                // Getting zeros
                arma::mat prediction_train_sum(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
                arma::mat prediction_test_sum(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);


                arma::vec partial_u(data.x_train.n_rows);
                arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
                arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);

                // Aux for used vars
                arma::cube cube_used_vars(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);

                // Iterating over the d-dimension MATRICES of the response.
                for(int j = 0; j < data.y_mat.n_cols; j++){


                        // Storing all used vars
                        arma::mat mat_used_vars(data.n_tree, data.x_train.n_cols);

                        // Covariance objects
                        arma::mat Sigma_j_mj(1,(data.y_mat.n_cols-1),arma::fill::zeros);
                        arma::mat Sigma_mj_j((data.y_mat.n_cols-1),1,arma::fill::zeros);
                        arma::mat Sigma_mj_mj = data.Sigma;


                        double Sigma_j_j = data.Sigma(j,j);

                        int aux_j_counter = 0;


                        // Dropping the column with respect to "j"
                        Sigma_mj_mj.shed_row(j);
                        Sigma_mj_mj.shed_col(j);

                        for(int d = 0; d < data.y_mat.n_cols; d++){

                                // Rcpp::Rcout <<  "AUX_J: " << data.Sigma(j,d) << endl;
                                if(d!=j){
                                        Sigma_j_mj(0,aux_j_counter)  = data.Sigma(j,d);
                                        Sigma_mj_j(aux_j_counter,0) = data.Sigma(j,d);
                                        aux_j_counter = aux_j_counter + 1;
                                }

                        }


                        // ============================================
                        // This step does not iterate over the trees!!!
                        // ===========================================
                        y_mj = data.y_mat;
                        y_mj.shed_col(j);
                        y_hat_mj = y_mat_hat;
                        y_hat_mj.shed_col(j);

                        // Calculating the invertion that gonna be used for the U and V
                        arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);
                        arma::mat Sigma_calculation_aux = (Sigma_mj_j.t()*Sigma_mj_mj_inv); // This line calculate a quantitiy that is constant for the i_train so avoids multiple calculations;

                        // Calculating the current partial U
                        for(int i_train = 0; i_train < data.y_mat.n_rows;i_train++){
                                partial_u(i_train) = arma::as_scalar(Sigma_calculation_aux*(y_mj.row(i_train)-y_hat_mj.row(i_train)).t()); // Old version

                        }

                        double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj*Sigma_mj_mj_inv*Sigma_mj_j);
                        data.v_j = v;


                        data.sigma_mu_j = data.sigma_mu(j);


                        // Updating the tree
                        for(int t = 0; t<data.n_tree;t++){

                                // Current tree counter
                                curr_tree_counter = t + j*data.n_tree;

                                // Creating the auxliar prediction vector
                                arma::vec y_j_hat(data.y_mat.n_rows,arma::fill::zeros);
                                arma::vec y_j_test_hat(data.x_test.n_rows,arma::fill::zeros);

                                // Updating the partial residuals
                                if(data.n_tree>1){
                                        partial_residuals = data.y_mat.col(j)-sum_exclude_col(tree_fits_store.slice(j),t);
                                } else {
                                        partial_residuals = data.y_mat.col(j);
                                }

                                // Iterating over all trees
                                verb = arma::randu(arma::distr_param(0.0,1.0));

                                if(all_forest.trees[curr_tree_counter]->isLeaf & all_forest.trees[curr_tree_counter]->isRoot){
                                        verb = 0.1;
                                }

                                // Selecting the verb
                                if(verb < 0.25){
                                        data.move_proposal(0)++;
                                        grow(all_forest.trees[curr_tree_counter],data,partial_residuals,partial_u,j);
                                } else if(verb>=0.25 & verb <0.5) {
                                        data.move_proposal(1)++;
                                        prune(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u);
                                } else {
                                        data.move_proposal(2)++;
                                        change(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u,j);
                                }


                                // Updating Mu and Prediction
                                updateMuPredictions(all_forest.trees[curr_tree_counter],data,y_j_hat,y_j_test_hat);

                                // Updating the tree
                                // cout << "Residuals error 2.0"<< endl;
                                tree_fits_store.slice(j).col(t) = y_j_hat;
                                // cout << "Residuals error 3.0"<< endl;
                                tree_fits_store_test.slice(j).col(t) = y_j_test_hat;
                                // cout << "Residuals error 4.0"<< endl;

                                // Aux for the used vars
                                if(var_selection_bool){
                                        arma::vec used_vars(data.x_train.n_cols);
                                        collect_split_vars(used_vars,all_forest.trees[curr_tree_counter]);
                                        mat_used_vars.row(t) = used_vars.t();
                                }
                                // arma::cout << "Used vars on tree: " << used_vars << endl;

                        } // End of iterations over "t"

                        // Summing over all trees
                        prediction_train_sum = sum(tree_fits_store.slice(j),1);
                        y_mat_hat.col(j) = prediction_train_sum;

                        prediction_test_sum = sum(tree_fits_store_test.slice(j),1);
                        y_mat_test_hat.col(j) = prediction_test_sum;

                        cube_used_vars.slice(j) = mat_used_vars;
                }// End of iterations over "j"


                // Storing cube of used vars
                all_j_tree_var(i) = cube_used_vars;

                // std::cout << "Error Tau: " << data.tau<< endl;
                if(update_Sigma){
                        // Updating or not a_j
                        if(hier_prior_bool){
                                update_a_j(data);
                        }
                        updateSigma(y_mat_hat, data);
                }

                all_Sigma_post.slice(i) = data.Sigma;

                // Updating the missing data;
                for (int ii = 0; ii < n_missing.size(); ii++){
                        if(n_missing(ii)==0){
                                continue;
                        } else {
                                update_y_mat_missing(data,
                                                     y_mat_hat,
                                                     na_indicators,
                                                     ii);
                        }
                }

                // std::cout << " All good " << endl;
                if(i >= n_burn){
                        // Storing the predictions
                        y_train_hat_post.slice(curr) = y_mat_hat;
                        y_test_hat_post.slice(curr) = y_mat_test_hat;
                        y_mat_post.slice(curr) = data.y_mat;
                        Sigma_post.slice(curr) = data.Sigma;
                        curr++;
                }

                pb += 1;

        }
        // Initialising PB
        Rcpp::Rcout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                Rcpp::Rcout << "=";
        }

        for(; k < width;k++){
                Rcpp::Rcout << " ";
        }

        Rcpp::Rcout << "] " << std::setprecision(5) << 100 << "%\r";
        Rcpp::Rcout.flush();

        Rcpp::Rcout << std::endl;

        return Rcpp::List::create(y_train_hat_post, //[1]
                                  y_test_hat_post, //[2]
                                  Sigma_post, //[3]
                                  all_Sigma_post, // [4]
                                  data.move_proposal, // [5]
                                  data.move_acceptance, //[6]
                                  all_j_tree_var, //[7]
                                  y_mat_post // [8]
        );
}


// =====================================
// CLASSIFICATION BART FUNCTIONS
// =====================================

//[[Rcpp::export]]
double truncated_sample(double mu, bool left, double sigma_) {

        double x;
        if(left){
                mu = -mu/sigma_;
        } else {
                mu = mu/sigma_;
        }
        double alpha = (mu + sqrt((mu) * (mu) + 4.0)) / 2.0;
        bool accept = false;
        int iteration_counter = 0;

        while (!accept) {
                double z = -log(arma::randu(arma::distr_param(0.0,1.0)))/alpha + (mu);
                double p;

                if (mu < alpha) {
                        p = exp(-((alpha - z) * (alpha - z)) / 2.0);
                } else {
                        p = exp(-((mu - alpha) * (mu - alpha)) / 2.0) * exp(-((alpha - z) * (alpha - z)) / 2.0);
                }

                double u = arma::randu(arma::distr_param(0.0,1.0));

                if (u < p) {

                        if(left){
                                x = z*sigma_ - mu;
                        } else {
                                x = -z*sigma_ + mu;
                        }
                        accept = true;
                }

                iteration_counter++;

                if(iteration_counter>1e6){
                        Rcpp::Rcout << "Mean value: " << mu << endl;
                        Rcpp::Rcout << "Sigma value: " << sigma_ << endl;

                        throw std::range_error("many iterations for the the truncated-sampler");
                }
        }


        return x;
}

double up_tn_sampler(arma::mat &z_mat_, arma::mat &mean_mat_, double lower, double v_j_,
                     int i_, int j_,
                     arma::mat &Sigma_mj_mj_inv_, arma::mat &Sigma_j_mj_,
                     arma::mat &Sigma_mj_j_, bool tn_sampler){

        bool sample_bool = true;
        int exit = 0;
        // bool tn_sampler = true;

        // Getting the shed vesion
        arma::mat z_mj = z_mat_;
        arma::mat z_mj_hat = mean_mat_;
        z_mj.shed_col(j_);
        z_mj_hat.shed_col(j_);
        double mean_ = mean_mat_(i_,j_) + arma::as_scalar((Sigma_mj_j_.t()*Sigma_mj_mj_inv_)*(z_mj.row(i_)-z_mj_hat.row(i_)).t()); // Old version

        if(tn_sampler){
                return truncated_sample(mean_,true,sqrt(v_j_));
        } else {
                while(sample_bool){

                        double sample = R::rnorm(mean_,sqrt(v_j_));

                        if(sample > lower){
                                return sample;
                        } else {
                                exit++;
                        }

                        if(exit > 1e8){
                                sample_bool =  false;
                        }
                }

                Rcpp::stop(" Choose another upper boundary");
                return 0.0;
        }
}




double lw_tn_sampler(arma::mat &z_mat_, arma::mat &mean_mat_, double upper, double v_j_,
                     int i_, int j_,
                     arma::mat &Sigma_mj_mj_inv_, arma::mat &Sigma_j_mj_,
                     arma::mat &Sigma_mj_j_, bool tn_sampler){

        bool sample_bool = true;
        int exit = 0;
        // bool tn_sampler =  true;


        // Getting the shed vesion
        arma::mat z_mj = z_mat_;
        arma::mat z_mj_hat = mean_mat_;
        z_mj.shed_col(j_);
        z_mj_hat.shed_col(j_);
        double mean_ = mean_mat_(i_,j_) + arma::as_scalar((Sigma_mj_j_.t()*Sigma_mj_mj_inv_)*(z_mj.row(i_)-z_mj_hat.row(i_)).t()); // Old version

        if(tn_sampler){
                return truncated_sample(mean_,false,sqrt(v_j_));
        } else {
                while(sample_bool){

                        double sample = R::rnorm(mean_,sqrt(v_j_));

                        if(sample <= upper){
                                return sample;
                        } else {
                                exit++;
                        }

                        if(exit > 1e8){
                                sample_bool =  false;
                        }
                }

                Rcpp::stop(" Choose another upper boundary");
                return 0.0;
        }
}

double z_missing_sampler(arma::mat &z_mat_, arma::mat &mean_mat_, double v_j_,
                         int i_, int j_,
                         arma::mat &Sigma_mj_mj_inv_, arma::mat &Sigma_j_mj_,
                         arma::mat &Sigma_mj_j_, bool tn_sampler){

     bool sample_bool = true;
     int exit = 0;
     // bool tn_sampler = true;

     // Getting the shed vesion
     arma::mat z_mj = z_mat_;
     arma::mat z_mj_hat = mean_mat_;
     z_mj.shed_col(j_);
     z_mj_hat.shed_col(j_);
     double mean_ = mean_mat_(i_,j_) + arma::as_scalar((Sigma_mj_j_.t()*Sigma_mj_mj_inv_)*(z_mj.row(i_)-z_mj_hat.row(i_)).t()); // Old version

     double sample = R::rnorm(mean_,sqrt(v_j_));

     return sample;
}



// Updating Z
void update_z(arma::mat &z_mat_,
              arma::mat &y_hat,
              modelParam &data,
              int j_,
              arma::mat &Sigma_mj_mj_inv_,
              arma::mat &Sigma_j_mj_,
              arma::mat &Sigma_mj_j_,
              bool tn_sampler){

        // cout << "Nrow z_mat_" << y_hat.n_rows << "-- ncols: " << y_hat.n_cols << endl;
        for(int i = 0; i < data.x_train.n_rows; i++){

             if(data.y_mat(i,j_)==1){
                        // cout << "Y_hat(" <<i<<","<<j_<<") :" << y_hat(i,j_) << endl;
                        z_mat_(i,j_) = up_tn_sampler(z_mat_,y_hat,0.0,data.v_j,
                               i,j_,Sigma_mj_mj_inv_,Sigma_j_mj_,Sigma_mj_j_,tn_sampler);
                } else if(data.y_mat(i,j_)==0){
                        z_mat_(i,j_) = lw_tn_sampler(z_mat_,y_hat,0.0,data.v_j,
                               i,j_,Sigma_mj_mj_inv_,Sigma_j_mj_,Sigma_mj_j_,tn_sampler);
                } else if (data.y_mat(i,j_)==-1){
                        z_mat_(i,j_) = z_missing_sampler(z_mat_,y_hat,data.v_j,
                              i,j_,Sigma_mj_mj_inv_,Sigma_j_mj_,Sigma_mj_j_,tn_sampler);
                } else {
                     Rcpp::Rcout << "Invalid outcome for Y" << endl;
                }
        }
}


// [[Rcpp::export]]
Rcpp::List cppbart_CLASS(arma::mat x_train,
                   arma::mat y_mat,
                   arma::mat x_test,
                   arma::mat x_cut,
                   int n_tree,
                   int node_min_size,
                   int n_mcmc,
                   int n_burn,
                   arma::mat Sigma_init,
                   arma::vec mu_init,
                   arma::vec sigma_mu,
                   double nu,
                   double alpha, double beta,
                   unsigned int m,
                   bool update_sigma,
                   bool var_selection_bool,
                   bool tn_sampler,
                   bool sv_bool,
                   arma::mat sv_matrix,
                   arma::vec categorical_indicators){

        // Posterior counter
        int curr = 0;

        // Creating a dummy object for the S_0 wish and A_j
        arma::mat S_0_wish = arma::mat(y_mat.n_cols,y_mat.n_cols,arma::fill::zeros);
        arma::vec A_j_vec = arma::vec(y_mat.n_cols,arma::fill::zeros);
        // cout << " Error on model.param" << endl;
        // Creating the structu object
        modelParam data(x_train,
                        y_mat,
                        x_test,
                        x_cut,
                        n_tree,
                        node_min_size,
                        alpha,
                        beta,
                        nu,
                        sigma_mu,
                        Sigma_init,
                        S_0_wish,
                        A_j_vec,
                        n_mcmc,
                        n_burn,
                        sv_bool,
                        sv_matrix,
                        categorical_indicators);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;

        // Rcpp::Rcout << "error here" << endl;

        // Defining those elements
        arma::cube y_train_hat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube y_test_hat_post(data.x_test.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_post,arma::fill::zeros);
        arma::cube all_Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_mcmc,arma::fill::zeros);
        arma::mat correlation_matrix_post(n_mcmc,0.5*data.Sigma.n_cols*(data.Sigma.n_cols-1));
        // Rcpp::Rcout << "error here2" << endl;

        // =====================================
        // For the moment I will not store those
        // =====================================
        // arma::cube all_tree_post(y_mat.size(),n_tree,n_post,arma::fill::zeros);


        // Defining other variables
        // arma::vec partial_pred = arma::mat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::vec partial_residuals(data.x_train.n_rows,arma::fill::zeros);
        arma::cube tree_fits_store(data.x_train.n_rows,data.n_tree,data.y_mat.n_cols,arma::fill::zeros);
        arma::cube tree_fits_store_test(data.x_test.n_rows,data.n_tree,y_mat.n_cols,arma::fill::zeros);


        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;


        // cout << " Error one " << endl;

        // Selecting the train
        Forest all_forest(data);

        // Creating variables to help define in which tree set we are;
        int curr_tree_counter;

        // Matrix that store all the predictions for all y
        arma::mat y_mat_hat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat y_mat_test_hat(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

        // Initialising the z_matrix
        arma::mat z_mat_train(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
        arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);

        // Setting a vector to store the variables used in the split
        arma::field<arma::cube> all_j_tree_var(data.n_mcmc);
        for(int cube_dim = 0; cube_dim <all_j_tree_var.size(); cube_dim ++){
                all_j_tree_var(cube_dim) = arma::cube(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);
        }

        for(int i = 0;i<data.n_mcmc;i++){

                // cout << "MCMC iter: " << i << endl;
                // Initialising PB
                Rcpp::Rcout << "[";
                int k = 0;
                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        Rcpp::Rcout << "=";
                }

                for(; k < width;k++){
                        Rcpp::Rcout << " ";
                }

                Rcpp::Rcout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                Rcpp::Rcout.flush();


                // Getting zeros
                arma::mat prediction_train_sum(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
                arma::mat prediction_test_sum(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

                // ==
                // This was here
                //===
                // arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
                // arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);
                //
                arma::vec partial_u(data.x_train.n_rows);


                // Aux for used vars
                arma::cube cube_used_vars(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);

                // Iterating over the d-dimension MATRICES of the response.
                for(int j = 0; j < data.y_mat.n_cols; j++){


                        // Storing all used vars
                        arma::mat mat_used_vars(data.n_tree, data.x_train.n_cols);

                        // cout << "Tree iter: " << j << endl;
                        // cout << "Sigma dim : " << data.Sigma.n_rows << " x " << data.Sigma.n_cols << endl;
                        arma::mat Sigma_j_mj(1,(data.y_mat.n_cols-1),arma::fill::zeros);
                        arma::mat Sigma_mj_j((data.y_mat.n_cols-1),1,arma::fill::zeros);
                        arma::mat Sigma_mj_mj = data.Sigma;


                        double Sigma_j_j = data.Sigma(j,j);

                        int aux_j_counter = 0;

                        // Dropping the column with respect to "j"
                        Sigma_mj_mj.shed_row(j);
                        Sigma_mj_mj.shed_col(j);

                        for(int d = 0; d < data.y_mat.n_cols; d++){

                                // Rcpp::Rcout <<  "AUX_J: " << data.Sigma(j,d) << endl;
                                if(d!=j){
                                        Sigma_j_mj(0,aux_j_counter)  = data.Sigma(j,d);
                                        Sigma_mj_j(aux_j_counter,0) = data.Sigma(j,d);
                                        aux_j_counter = aux_j_counter + 1;
                                }

                        }

                        // cout << " Error here!" << endl;
                        // ============================================
                        // This step does not iterate over the trees!!!
                        // ===========================================
                        y_mj = z_mat_train; // PAY ATTENTION THAT HERE I USING Z_MAT INSTEAD!
                        y_mj.shed_col(j);
                        y_hat_mj = y_mat_hat;
                        y_hat_mj.shed_col(j);

                        // Calculating the invertion that gonna be used for the U and V
                        arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);

                        // Calculating the current partial U
                        for(int i_train = 0; i_train < data.y_mat.n_rows;i_train++){
                                // cout << "The scale factor of  the residuals " << Sigma_mj_j*Sigma_mj_mj_inv <<endl;
                                partial_u(i_train) = arma::as_scalar((Sigma_mj_j.t()*Sigma_mj_mj_inv)*(y_mj.row(i_train)-y_hat_mj.row(i_train)).t()); // Old version
                                // cout << "error here" << endl;
                        }

                        double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj*(Sigma_mj_mj_inv*Sigma_mj_j));
                        data.v_j = v;
                        // cout << "Sigma_jj: " << Sigma_mj_mj_inv<< endl;
                        // cout << " Variance term: " << data.R(0,1) << endl;

                        data.sigma_mu_j = data.sigma_mu(j);

                        // cout << " Error here! 2.0" << endl;

                        // Updating the tree
                        for(int t = 0; t<data.n_tree;t++){

                                // Current tree counter
                                curr_tree_counter = t + j*data.n_tree;
                                // cout << "curr_tree_counter value:" << curr_tree_counter << endl;
                                // Creating the auxliar prediction vector
                                arma::vec y_j_hat(data.y_mat.n_rows,arma::fill::zeros);
                                arma::vec y_j_test_hat(data.x_test.n_rows,arma::fill::zeros);

                                // Updating the partial residuals
                                if(data.n_tree>1){
                                        partial_residuals = z_mat_train.col(j)-sum_exclude_col(tree_fits_store.slice(j),t);
                                } else {
                                        partial_residuals = z_mat_train.col(j);
                                }

                                // Iterating over all trees
                                verb = arma::randu(arma::distr_param(0.0,1.0));

                                if(all_forest.trees[curr_tree_counter]->isLeaf & all_forest.trees[curr_tree_counter]->isRoot){
                                        // verb = arma::randu(arma::distr_param(0.0,0.3));
                                        verb = 0.1;
                                }

                                // Selecting the verb
                                if(verb < 0.25){
                                        data.move_proposal(0)++;
                                        // cout << " Grow error" << endl;
                                        // Rcpp::stop("STOP ENTERED INTO A GROW");
                                        grow(all_forest.trees[curr_tree_counter],data,partial_residuals,partial_u,j);
                                } else if(verb>=0.25 & verb <0.5) {
                                        data.move_proposal(1)++;
                                        prune(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u);
                                } else {
                                        data.move_proposal(2)++;
                                        change(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u,j);
                                }


                                updateMu(all_forest.trees[curr_tree_counter],data,partial_residuals,partial_u);

                                // Getting predictions
                                // cout << " Error on Get Predictions" << endl;
                                getPredictions(all_forest.trees[curr_tree_counter],data,y_j_hat,y_j_test_hat);

                                // Updating the tree
                                // cout << "Residuals error 2.0"<< endl;
                                tree_fits_store.slice(j).col(t) = y_j_hat;
                                // cout << "Residuals error 3.0"<< endl;
                                tree_fits_store_test.slice(j).col(t) = y_j_test_hat;
                                // cout << "Residuals error 4.0"<< endl;

                                // Aux for the used vars
                                if(var_selection_bool){
                                        arma::vec used_vars(data.x_train.n_cols);
                                        collect_split_vars(used_vars,all_forest.trees[curr_tree_counter]);
                                        mat_used_vars.row(t) = used_vars.t();
                                }

                        } // End of iterations over "t"


                        // Storing all used VARS for the y_{j} dimension
                        cube_used_vars.slice(j) = mat_used_vars;


                        // Summing over all trees
                        prediction_train_sum = sum(tree_fits_store.slice(j),1);
                        y_mat_hat.col(j) = prediction_train_sum;

                        prediction_test_sum = sum(tree_fits_store_test.slice(j),1);
                        y_mat_test_hat.col(j) = prediction_test_sum;

                        // Updating z_j values
                        // Rcpp::Rcout << "Error on update z" << endl;
                        update_z(z_mat_train,y_mat_hat,data,j,Sigma_mj_mj_inv,Sigma_j_mj,Sigma_mj_j,tn_sampler);
                        // Rcpp::Rcout << "Sucess!" << endl;

                }// End of iterations over "j"


                // Storing cube of used vars
                all_j_tree_var(i) = cube_used_vars;


                if(update_sigma){
                        arma::mat residuals_ = y_mat_hat-z_mat_train;
                        // arma::mat residuals_ = generateZ(y_mat.n_rows,y_mat.n_cols,Sigma_init);
                        // arma::mat residuals_ = z_mat_train;
                        // Updating the Sigma
                        // arma::cout  << " Error here on the first update:" << m << endl;
                        // arma::cout << " R dimensions " << data.R.n_rows << "x" << data.R.n_cols << arma::endl;
                        arma::mat D_sqrt = sqrt(data.D);
                        data.W = D_sqrt*data.R*D_sqrt;
                        arma::mat W_proposal = arma::iwishrnd(m*data.W,m);
                        arma::mat D_proposal = arma::diagmat(W_proposal);
                        // arma::cout << D_proposal << endl;
                        // arma::cout << " D proposal dimensions " << D_proposal.n_rows << "x" << D_proposal.n_cols << arma::endl;
                        arma::mat inv_D_sqrt  = arma::inv(sqrt(D_proposal));
                        arma::mat R_proposal = inv_D_sqrt*W_proposal*inv_D_sqrt;
                        // arma::cout << R_proposal << endl;
                        // R_proposal.diag() = arma::ones(data.R.n_cols);
                        // if(R_proposal(0,0) != 1.0){
                        //         arma::cout << "CORRELATION VALUE: "<< std::setprecision(10) << (R_proposal(0,0) - 1.0) << std::endl;
                        //         throw std::range_error("Incorrect correlation matrix");
                        // }
                        // cout << "Part one: " << log_posterior_dens(R_proposal,D_proposal,nu,residuals_,false) << endl;
                        // cout << "Part two: " << log_posterior_dens(data.R,data.D,nu,residuals_,false) << endl;
                        // cout << "Part three: " << log_proposal_dens(data.R,data.D,nu,R_proposal,D_proposal,m) << endl;
                        // cout << "Part four: " << log_proposal_dens(R_proposal,D_proposal,nu,data.R,data.D,m) << endl;

                        double alpha_corr = exp(log_posterior_dens(R_proposal,D_proposal,nu,residuals_,false) -
                                                log_posterior_dens(data.R,data.D,nu,residuals_,false) +
                                                log_proposal_dens(data.R,data.D,nu,R_proposal,D_proposal,m) -
                                                log_proposal_dens(R_proposal,D_proposal,nu,data.R,data.D,m));
                        double unif_sample = arma::randu(arma::distr_param(0.0,1.0));
                        // cout << "Uniform sample is: " << unif_sample << endl;
                        // cout << "Alpha sample is: " << alpha_corr << endl;

                        if( unif_sample < alpha_corr) {
                                data.R = R_proposal;
                                data.D = D_proposal;
                                // arma::cout << "YEEE" << endl;
                                if(update_sigma){
                                        data.Sigma = R_proposal;
                                }
                                // arma::cout << " Expressing Sigma(i,i): " << R_proposal.diag() << endl;
                        }  else {
                                if(update_sigma){
                                        data.Sigma = data.R;
                                }
                        }
                }

                all_Sigma_post.slice(i) = data.R;

                // std::cout << " All good " << endl;
                if(i >= n_burn){
                        // Storing the predictions
                        y_train_hat_post.slice(curr) = y_mat_hat;
                        y_test_hat_post.slice(curr) = y_mat_test_hat;
                        Sigma_post.slice(curr) = data.R;
                        correlation_matrix_post.row(curr) = (makeSigmaInv(data.R)).t();
                        curr++;
                }

                pb += 1;

        }
        // Initialising PB
        Rcpp::Rcout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                Rcpp::Rcout << "=";
        }

        for(; k < width;k++){
                Rcpp::Rcout << " ";
        }

        Rcpp::Rcout << "] " << std::setprecision(5) << 100 << "%\r";
        Rcpp::Rcout.flush();

        Rcpp::Rcout << std::endl;

        return Rcpp::List::create(y_train_hat_post, //[1]
                                  y_test_hat_post, //[2]
                                  Sigma_post, //[3]
                                  all_Sigma_post, // [4]
                                  data.move_proposal, // [5]
                                  data.move_acceptance, //[6]
                                  correlation_matrix_post, // [7]
                                  all_j_tree_var // [[8]]
        );
}


// [[Rcpp::export]]
Rcpp::List cppbart_univariate_CLASS(arma::mat x_train,
                                arma::mat y_mat,
                                arma::mat x_test,
                                arma::mat x_cut,
                                int n_tree,
                                int node_min_size,
                                int n_mcmc,
                                int n_burn,
                                arma::mat Sigma_init,
                                arma::vec mu_init,
                                arma::vec sigma_mu,
                                double nu,
                                double alpha, double beta,
                                unsigned int m,
                                bool update_sigma,
                                bool var_selection_bool,
                                bool tn_sampler,
                                bool sv_bool,
                                arma::mat sv_matrix,
                                arma::vec categorical_indicators){

        // Posterior counter
        int curr = 0;

        // Creating a dummy object for the S_0 wish and A_j
        arma::mat S_0_wish = arma::mat(y_mat.n_cols,y_mat.n_cols,arma::fill::zeros);
        arma::vec A_j_vec = arma::vec(y_mat.n_cols,arma::fill::zeros);
        // cout << " Error on model.param" << endl;
        // Creating the structu object
        modelParam data(x_train,
                        y_mat,
                        x_test,
                        x_cut,
                        n_tree,
                        node_min_size,
                        alpha,
                        beta,
                        nu,
                        sigma_mu,
                        Sigma_init,
                        S_0_wish,
                        A_j_vec,
                        n_mcmc,
                        n_burn,
                        sv_bool,
                        sv_matrix,
                        categorical_indicators);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;

        // Rcpp::Rcout << "error here" << endl;

        // Defining those elements
        arma::cube y_train_hat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube y_test_hat_post(data.x_test.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_post,arma::fill::zeros);
        arma::cube all_Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_mcmc,arma::fill::zeros);
        arma::mat correlation_matrix_post(n_mcmc,0.5*data.Sigma.n_cols*(data.Sigma.n_cols-1));
        // Rcpp::Rcout << "error here2" << endl;

        // =====================================
        // For the moment I will not store those
        // =====================================
        // arma::cube all_tree_post(y_mat.size(),n_tree,n_post,arma::fill::zeros);


        // Defining other variables
        // arma::vec partial_pred = arma::mat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::vec partial_residuals(data.x_train.n_rows,arma::fill::zeros);
        arma::cube tree_fits_store(data.x_train.n_rows,data.n_tree,data.y_mat.n_cols,arma::fill::zeros);
        arma::cube tree_fits_store_test(data.x_test.n_rows,data.n_tree,y_mat.n_cols,arma::fill::zeros);


        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;


        // cout << " Error one " << endl;

        // Selecting the train
        Forest all_forest(data);

        // Creating variables to help define in which tree set we are;
        int curr_tree_counter;

        // Matrix that store all the predictions for all y
        arma::mat y_mat_hat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat y_mat_test_hat(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

        // Initialising the z_matrix
        arma::mat z_mat_train(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
        arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);

        // Setting a vector to store the variables used in the split
        arma::field<arma::cube> all_j_tree_var(data.n_mcmc);
        for(int cube_dim = 0; cube_dim <all_j_tree_var.size(); cube_dim ++){
                all_j_tree_var(cube_dim) = arma::cube(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);
        }

        for(int i = 0;i<data.n_mcmc;i++){

                // cout << "MCMC iter: " << i << endl;
                // Initialising PB
                Rcpp::Rcout << "[";
                int k = 0;
                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        Rcpp::Rcout << "=";
                }

                for(; k < width;k++){
                        Rcpp::Rcout << " ";
                }

                Rcpp::Rcout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                Rcpp::Rcout.flush();


                // Getting zeros
                arma::mat prediction_train_sum(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
                arma::mat prediction_test_sum(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

                // ==
                // This was here
                //===
                // arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
                // arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);
                //
                arma::vec partial_u(data.x_train.n_rows);


                // Aux for used vars
                arma::cube cube_used_vars(data.n_tree,data.x_train.n_cols,data.y_mat.n_cols);

                // Iterating over the d-dimension MATRICES of the response.
                for(int j = 0; j < data.y_mat.n_cols; j++){


                        // Storing all used vars
                        arma::mat mat_used_vars(data.n_tree, data.x_train.n_cols);

                        // cout << "Tree iter: " << j << endl;
                        // cout << "Sigma dim : " << data.Sigma.n_rows << " x " << data.Sigma.n_cols << endl;
                        arma::mat Sigma_j_mj(1,(data.y_mat.n_cols-1),arma::fill::zeros);
                        arma::mat Sigma_mj_j((data.y_mat.n_cols-1),1,arma::fill::zeros);
                        arma::mat Sigma_mj_mj = data.Sigma;


                        double Sigma_j_j = data.Sigma(j,j);

                        int aux_j_counter = 0;

                        // Dropping the column with respect to "j"
                        Sigma_mj_mj.shed_row(j);
                        Sigma_mj_mj.shed_col(j);

                        for(int d = 0; d < data.y_mat.n_cols; d++){

                                // Rcpp::Rcout <<  "AUX_J: " << data.Sigma(j,d) << endl;
                                if(d!=j){
                                        Sigma_j_mj(0,aux_j_counter)  = data.Sigma(j,d);
                                        Sigma_mj_j(aux_j_counter,0) = data.Sigma(j,d);
                                        aux_j_counter = aux_j_counter + 1;
                                }

                        }

                        // cout << " Error here!" << endl;
                        // ============================================
                        // This step does not iterate over the trees!!!
                        // ===========================================
                        y_mj = z_mat_train; // PAY ATTENTION THAT HERE I USING Z_MAT INSTEAD!
                        y_mj.shed_col(j);
                        y_hat_mj = y_mat_hat;
                        y_hat_mj.shed_col(j);

                        // Calculating the invertion that gonna be used for the U and V
                        arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);

                        // Calculating the current partial U
                        for(int i_train = 0; i_train < data.y_mat.n_rows;i_train++){
                                // cout << "The scale factor of  the residuals " << Sigma_mj_j*Sigma_mj_mj_inv <<endl;
                                partial_u(i_train) = arma::as_scalar((Sigma_mj_j.t()*Sigma_mj_mj_inv)*(y_mj.row(i_train)-y_hat_mj.row(i_train)).t()); // Old version
                                // cout << "error here" << endl;
                        }

                        double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj*(Sigma_mj_mj_inv*Sigma_mj_j));
                        data.v_j = v;
                        // cout << "Sigma_jj: " << Sigma_mj_mj_inv<< endl;
                        // cout << " Variance term: " << data.R(0,1) << endl;

                        data.sigma_mu_j = data.sigma_mu(j);

                        // cout << " Error here! 2.0" << endl;

                        // Updating the tree
                        for(int t = 0; t<data.n_tree;t++){

                                // Current tree counter
                                curr_tree_counter = t + j*data.n_tree;
                                // cout << "curr_tree_counter value:" << curr_tree_counter << endl;
                                // Creating the auxliar prediction vector
                                arma::vec y_j_hat(data.y_mat.n_rows,arma::fill::zeros);
                                arma::vec y_j_test_hat(data.x_test.n_rows,arma::fill::zeros);

                                // Updating the partial residuals
                                if(data.n_tree>1){
                                        partial_residuals = z_mat_train.col(j)-sum_exclude_col(tree_fits_store.slice(j),t);
                                } else {
                                        partial_residuals = z_mat_train.col(j);
                                }

                                // Iterating over all trees
                                verb = arma::randu(arma::distr_param(0.0,1.0));

                                if(all_forest.trees[curr_tree_counter]->isLeaf & all_forest.trees[curr_tree_counter]->isRoot){
                                        // verb = arma::randu(arma::distr_param(0.0,0.3));
                                        verb = 0.1;
                                }

                                // Selecting the verb
                                if(verb < 0.25){
                                        data.move_proposal(0)++;
                                        // cout << " Grow error" << endl;
                                        // Rcpp::stop("STOP ENTERED INTO A GROW");
                                        grow_uni(all_forest.trees[curr_tree_counter],data,partial_residuals,j);
                                } else if(verb>=0.25 & verb <0.5) {
                                        data.move_proposal(1)++;
                                        prune_uni(all_forest.trees[curr_tree_counter], data, partial_residuals);
                                } else {
                                        data.move_proposal(2)++;
                                        change_uni(all_forest.trees[curr_tree_counter], data, partial_residuals,j);
                                }


                                updateMu(all_forest.trees[curr_tree_counter],data,partial_residuals,partial_u);

                                // Getting predictions
                                // cout << " Error on Get Predictions" << endl;
                                getPredictions(all_forest.trees[curr_tree_counter],data,y_j_hat,y_j_test_hat);

                                // Updating the tree
                                // cout << "Residuals error 2.0"<< endl;
                                tree_fits_store.slice(j).col(t) = y_j_hat;
                                // cout << "Residuals error 3.0"<< endl;
                                tree_fits_store_test.slice(j).col(t) = y_j_test_hat;
                                // cout << "Residuals error 4.0"<< endl;

                                // Aux for the used vars
                                if(var_selection_bool){
                                        arma::vec used_vars(data.x_train.n_cols);
                                        collect_split_vars(used_vars,all_forest.trees[curr_tree_counter]);
                                        mat_used_vars.row(t) = used_vars.t();
                                }

                        } // End of iterations over "t"


                        // Storing all used VARS for the y_{j} dimension
                        cube_used_vars.slice(j) = mat_used_vars;


                        // Summing over all trees
                        prediction_train_sum = sum(tree_fits_store.slice(j),1);
                        y_mat_hat.col(j) = prediction_train_sum;

                        prediction_test_sum = sum(tree_fits_store_test.slice(j),1);
                        y_mat_test_hat.col(j) = prediction_test_sum;

                        // Updating z_j values
                        // Rcpp::Rcout << "Error on update z" << endl;
                        update_z(z_mat_train,y_mat_hat,data,j,Sigma_mj_mj_inv,Sigma_j_mj,Sigma_mj_j,tn_sampler);
                        // Rcpp::Rcout << "Sucess!" << endl;

                }// End of iterations over "j"


                // Storing cube of used vars
                all_j_tree_var(i) = cube_used_vars;

                all_Sigma_post.slice(i) = data.R;

                if(i >= n_burn){
                        // Storing the predictions
                        y_train_hat_post.slice(curr) = y_mat_hat;
                        y_test_hat_post.slice(curr) = y_mat_test_hat;
                        Sigma_post.slice(curr) = data.R;
                        curr++;
                }

                pb += 1;

        }
        // Initialising PB
        Rcpp::Rcout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                Rcpp::Rcout << "=";
        }

        for(; k < width;k++){
                Rcpp::Rcout << " ";
        }

        Rcpp::Rcout << "] " << std::setprecision(5) << 100 << "%\r";
        Rcpp::Rcout.flush();

        Rcpp::Rcout << std::endl;

        return Rcpp::List::create(y_train_hat_post, //[1]
                                  y_test_hat_post, //[2]
                                  Sigma_post, //[3]
                                  all_Sigma_post, // [4]
                                  data.move_proposal, // [5]
                                  data.move_acceptance, //[6]
                                  correlation_matrix_post, // [7]
                                  all_j_tree_var // [[8]]
        );
}
