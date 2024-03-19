import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator as glm
from h2o.exceptions import H2OValueError
from h2o.grid.grid_search import H2OGridSearch

def gen_constraint_glm_model(training_dataset, x, y, solver="AUTO", family="gaussian", linear_constraints=None, 
                             beta_constraints=None, separate_linear_beta=False, init_optimal_glm=False, startval=None,
                             constraint_eta0=0.1258925, constraint_tau=10, constraint_alpha=0.1, 
                             constraint_beta=0.9, constraint_c0=10):
    """
    This function given the parameters will return a constraint GLM model.
    """
    if linear_constraints is None:
        raise H2OValueError("linear_constraints cannot be None")
        
    params = {"family":family, "lambda_":0.0, "seed":12345, "remove_collinear_columns":True, "solver":solver, 
              "linear_constraints":linear_constraints, "init_optimal_glm":init_optimal_glm, 
              "constraint_eta0":constraint_eta0, "constraint_tau":constraint_tau, "constraint_alpha":constraint_alpha,
              "constraint_beta":constraint_beta, "constraint_c0":constraint_c0}
    if beta_constraints is not None:
        params['beta_constraints']=beta_constraints
        params["separate_linear_beta"]=separate_linear_beta
    if startval is not None:
        params["startval"]=startval
        
    constraint_glm = glm(**params)
    constraint_glm.train(x=x, y=y, training_frame=training_dataset)
    return constraint_glm

def constraint_glm_gridsearch(training_dataset, x, y, solver="AUTO", family="gaussia", linear_constraints=None,
                              beta_constraints=None, metric="logloss", return_best=True, startval=None, 
                              init_optimal_glm=False, constraint_eta0=[0.1258925],  constraint_tau=[10], 
                              constraint_alpha=[0.1], constraint_beta=[0.9], constraint_c0=[10]):
    """
    This function given the obj_eps_hyper and inner_loop_hyper will build and run a gridsearch model and return the one
    with the best metric.
    """
    if linear_constraints is None:
        raise H2OValueError("linear_constraints cannot be None")

    params = {"family":family, "lambda_":0.0, "seed":12345, "remove_collinear_columns":True, "solver":solver,
              "linear_constraints":linear_constraints}
    hyperParams = {"constraint_eta0":constraint_eta0, "constraint_tau":constraint_tau, "constraint_alpha":constraint_alpha,
                   "constraint_beta":constraint_beta, "constraint_c0":constraint_c0}
    if beta_constraints is not None:
        params['beta_constraints']=beta_constraints
        hyperParams["separate_linear_beta"] = [True, False]
    if startval is not None:
        params["startval"]=startval
    if init_optimal_glm:
        params["init_optimal_glm"]=True
        
    glmGrid = H2OGridSearch(glm(**params), hyper_params=hyperParams)
    glmGrid.train(x=x, y=y, training_frame=training_dataset)
    sortedGrid = glmGrid.get_grid()
    print(sortedGrid)
    if return_best:
        return sortedGrid.models[0]
    else:
        return grid_models_analysis(sortedGrid.models, metric=metric)

def grid_models_analysis(grid_models, metric="logloss", epsilon=1e-3):
    """
    This method will search within the grid search models that have metrics within epsilon calculated as 
    abs(metric1-metric2)/abs(metric1) as the best model.  We are wanting to send the best model that has the lowerest
    equality constraint if it exists.  Else, the original top model will be returned.
    """
    base_metric = grid_models[0].model_performance()._metric_json[metric]
    base_constraints_table = grid_models[0]._model_json["output"]["linear_constraints_table"]
    num_constraints = len(base_constraints_table.cell_values)
    equality_exist = False
    cond_index = base_constraints_table.col_header.index("condition")
    num_equality = 0
    base_equality_constraints=[]
    for ind in range(num_constraints):
        if base_constraints_table.cell_values[ind][cond_index] == "== 0":
            equality_exist=True
            num_equality = num_equality+1
            base_equality_constraints.append(base_constraints_table.cell_values[ind][cond_index-1])

    if not(equality_exist):
        return grid_models[0]
    num_models = len(grid_models)
    best_model_ind = 0
    model_indices = []
    model_equality_constraints_values = []
    for ind in range(1, num_models):
        curr_model = grid_models[ind]
        curr_metric = grid_models[ind].model_performance()._metric_json[metric]
        metric_diff = abs(base_metric-curr_metric)/abs(base_metric)
        if metric_diff < epsilon:
            curr_constraint_table = curr_model._model_json["output"]["linear_constraints_table"]
            equality_constraints_values = []
            for ind2 in range(0, num_constraints): # collect all equality constraint info
                if curr_constraint_table.cell_values[ind2][cond_index]=="== 0":
                    equality_constraints_values.append(curr_constraint_table.cell_values[ind2][cond_index-1])
            # compare current equality and base equality constraint and choose the one with smallest magnitude
            better_model = compare_tuple(base_equality_constraints, equality_constraints_values)
            if better_model:
                best_model_ind = ind
                base_equality_constraints=equality_constraints_values
            model_equality_constraints_values.append(equality_constraints_values)
            model_indices.append(ind)
    print("best equality constraint values: {0} and it is from model index: {1}".format(base_equality_constraints, best_model_ind))
    return grid_models[best_model_ind]

def compare_tuple(original_tuple, new_tuple):
    """
    This function will return True if new_tuple has smaller magnitude elements than what is in original_tuple.
    """
    num_ele = len(original_tuple)
    assert num_ele==len(new_tuple)
    for ind in range(num_ele):
        if abs(original_tuple[ind]) <= abs(new_tuple[ind]):
            return False
    return True
    
       
def find_glm_iterations(glm_model):
    """
    Given a glm constrainted model, this method will obtain the number of iterations from the model summary.
    """
    cell_values = glm_model._model_json["output"]["model_summary"].cell_values
    lengths = len(cell_values)
    iteration_index = glm_model._model_json["output"]["model_summary"].col_header.index("number_of_iterations")
    return cell_values[lengths-1][iteration_index]
    
