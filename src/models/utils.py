
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def register_layers(model, name_list, prefix):
    for i, idx in enumerate(name_list):
        model[idx].register_forward_hook(get_activation(prefix+'_{}'.format(i)))
    #return model
def get_student_features(name_list, prefix):
    data = []
    for name in name_list:
        data.append(activation[prefix+'_{}'.format(name)])
    return data