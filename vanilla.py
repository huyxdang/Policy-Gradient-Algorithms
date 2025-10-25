# Implementation of a vanilla policy gradient algorithm

def create_model(number_of_observations: int, number_of_actions: int):
    """ Create an 1-hidden-layer MLP with ReLu actionvation function

    Returns: 
        nn.Module: Simple MLP model with 1 hidden layer
    """
    hidden_layer_features = 32
    model = nn.Sequential(
        nn.Linear(int_features = number_of_observations, 
            out_features = hidden_layer_features, bias = True),
        nn.ReLU(),
        nn.Linear(int_features = hidden_layer_features, 
        out_features = number_of_actions, bias = True)
    )
    return model

