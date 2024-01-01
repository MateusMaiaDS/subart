#include<RcppArmadillo.h>
#include<vector>
// Creating the struct
struct Node;
struct modelParam;

struct modelParam {

        arma::mat x_train;
        arma::mat y_mat;
        arma::mat x_test;
        arma::mat xcut;


        // BART prior param specification
        int n_tree;
        int d_var; // Dimension of variables in my base
        double alpha;
        double beta;
        arma::vec sigma_mu;
        arma::mat Sigma;
        arma::mat S_0_wish;
        arma::vec a_j_vec;
        arma::vec A_j_vec;
        arma::mat W;
        arma::mat R;
        arma::mat D;

        // Specific variables for each tree
        bool sv_bool;
        arma::mat sv_matrix;

        double nu;
        int node_min_size;

        // MCMC spec.
        int n_mcmc;
        int n_burn;

        // Create an indicator of accepted grown trees
        arma::vec move_proposal;
        arma::vec move_acceptance;

        // Elements to be used in the loglikelihood update and mu update
        double v_j;
        double sigma_mu_j;

        // Defining the constructor for the model param
        modelParam(arma::mat x_train_,
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
                   arma::mat sv_matrix_);

};

// Creating a forest
class Forest {

public:
        std::vector<Node*> trees;

        Forest(modelParam &data);
        // ~Forest();
};



// Creating the node struct
struct Node {

     bool isRoot;
     bool isLeaf;
     Node* left;
     Node* right;
     Node* parent;
     arma::vec train_index;
     arma::vec test_index;

     // Branch parameters
     int var_split;
     double var_split_rule;
     double lower;
     double upper;
     double curr_weight; // indicates if the observation is within terminal node or not
     int depth = 0;


     // Leaf parameters
     double mu;

     // Storing sufficient statistics over the nodes
     double log_likelihood = 0.0;
     double r_sum = 0.0;
     double u_sum = 0.0;
     double Gamma_j;
     double sigma_mu_j_sq;
     double S_j;


     int n_leaf = 0;
     int n_leaf_test = 0;

     // Creating the methods
     void addingLeaves(modelParam& data);
     void deletingLeaves();
     void Stump(modelParam& data);
     void updateWeight(const arma::mat X, int i);
     void getLimits(); // This function will get previous limit for the current var
     bool isLeft();
     bool isRight();
     void grow(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec &curr_u);
     void prune(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u);
     void change(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u);

     void nodeLogLike(modelParam &data);
     void updateResiduals(modelParam& data, arma::vec &curr_res, arma::vec &curr_u);
     void displayCurrNode();

     Node(modelParam &data);
     ~Node();
};

// Creating a function to get the leaves
void leaves(Node* x, std::vector<Node*>& leaves); // This function gonna modify by address the vector of leaves
std::vector<Node*> leaves(Node*x);

