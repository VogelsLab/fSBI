import subprocess
import numpy as np
import os
import time

def generate_call_auryn_bg_TIF_IE_6pPol(args):
    rule_str = " --rule a" + str(args["tau_pre"]) + "a" + str(args["tau_post"]) + "a"+\
            str(args["alpha"]) +  "a" + str(args["beta"]) + "a" + str(args["gamma"]) +\
            "a" + str(args["kappa"]) + "a"
    
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                rule_str +\
                " --wmax " + str(args["wmax"]) +\
                " --eta " + str(args["eta"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --NE " + str(args["NE"]) +\
                " --NI " + str(args["NI"]) +\
                " --N_input " + str(args["N_input"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def generate_call_auryn_bg_IF_IE_6pPol(args):
    rule_str = " --rule a" + str(args["tau_pre"]) + "a" + str(args["tau_post"]) + "a"+\
            str(args["alpha"]) +  "a" + str(args["beta"]) + "a" + str(args["gamma"]) +\
            "a" + str(args["kappa"]) + "a"
    
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                rule_str +\
                " --wmax " + str(args["wmax"]) +\
                " --eta " + str(args["eta"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --NE " + str(args["NE"]) +\
                " --NI " + str(args["NI"]) +\
                " --tau_ampa " + str(args["tau_ampa"]) +\
                " --tau_gaba " + str(args["tau_gaba"]) +\
                " --tau_nmda " + str(args["tau_nmda"]) +\
                " --ampa_nmda_ratio " + str(args["ampa_nmda_ratio"]) +\
                " --N_input " + str(args["N_input"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def generate_call_auryn_bg_AdEx_IE_6pPol(args):
    rule_str = " --rule a" + str(args["tau_pre"]) + "a" + str(args["tau_post"]) + "a"+\
            str(args["alpha"]) +  "a" + str(args["beta"]) + "a" + str(args["gamma"]) +\
            "a" + str(args["kappa"]) + "a"
    
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                rule_str +\
                " --wmax " + str(args["wmax"]) +\
                " --eta " + str(args["eta"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --NE " + str(args["NE"]) +\
                " --NI " + str(args["NI"]) +\
                " --N_input " + str(args["N_input"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def generate_call_auryn_bg_TIF_EEIE_TwvcPol(args):
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                " --rule_EE " + str(args["rule_EE"]) +\
                " --rule_IE " + str(args["rule_IE"]) +\
                " --wmax " + str(args["wmax"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def generate_call_auryn_bg_TIF_IE_TwvcMLP(args):
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                " --rule_IE " + str(args["rule_IE"]) +\
                " --eta " + str(args["eta"]) +\
                " --wmax " + str(args["wmax"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def generate_call_auryn_bg_IF_IE_TwvcMLP(args):
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                " --rule_IE " + str(args["rule_IE"]) +\
                " --eta " + str(args["eta"]) +\
                " --wmax " + str(args["wmax"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --NE " + str(args["NE"]) +\
                " --NI " + str(args["NI"]) +\
                " --N_input " + str(args["N_input"]) +\
                " --tau_ampa " + str(args["tau_ampa"]) +\
                " --tau_gaba " + str(args["tau_gaba"]) +\
                " --tau_nmda " + str(args["tau_nmda"]) +\
                " --ampa_nmda_ratio " + str(args["ampa_nmda_ratio"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def generate_call_auryn_bg_IF_EEEIIEII_6pPol(args):
    cl_str = args["auryn_sim_dir"] + str(args["name"]) + " --ID " + str(args["id"]) +\
                args["rule_str"] +\
                " --wmax " + str(args["wmax"]) +\
                " --eta " + str(args["eta"]) +\
                " --wee " + str(args["wee"]) +\
                " --wei " + str(args["wei"]) +\
                " --wie " + str(args["wie"]) +\
                " --wii " + str(args["wii"]) +\
                " --sparseness " + str(args["sparseness"]) +\
                " --NE " + str(args["NE"]) +\
                " --NI " + str(args["NI"]) +\
                " --tau_ampa " + str(args["tau_ampa"]) +\
                " --tau_gaba " + str(args["tau_gaba"]) +\
                " --tau_nmda " + str(args["tau_nmda"]) +\
                " --ampa_nmda_ratio " + str(args["ampa_nmda_ratio"]) +\
                " --N_input " + str(args["N_input"]) +\
                " --sparseness_poisson " + str(args["sparseness_poisson"]) +\
                " --rate_poisson " + str(args["rate_poisson"]) +\
                " --weight_poisson " + str(args["weight_poisson"]) +\
                " --max_rate_checker " + str(args["max_rate_checker"]) +\
                " --tau_checker " + str(args["tau_checker"]) +\
                " --lns " + str(args["length_training"]) +\
                " --ls " + str(args["length_scoring"]) +\
                " --workdir " + args["workdir"]
    return(cl_str)

def parse_saved_rule_TwvcMLP(A):
    """
    Assumes we are getting a 56 float vector from SpikES
    returns (taus, W1, W3_pre, W3_post, W4_pre, W4_post) that can be passed to other functions here
    """
    taus = np.zeros(4)
    W1 = np.zeros((8,4))
    W3_pre = np.zeros((4,2))
    W3_post = np.zeros((4,2))
    W4_pre = np.zeros((2,1))
    W4_post = np.zeros((2,1))
    ct = 0
    for i in range(len(taus)):
        taus[i] = A[ct]
        ct += 1
    # for i in range (len(W1)):
    #     for j in range(len(W1[0])):
    #         W1[i,j] = A[ct]
    #         ct += 1
    # for i in range (len(W3_pre)):
    #     for j in range(len(W3_pre[0])):
    #         W3_pre[i,j] = A[ct]
    #         ct += 1
    # for i in range (len(W3_post)):
    #     for j in range(len(W3_post[0])):
    #         W3_post[i,j] = A[ct]
    #         ct += 1
    # for i in range (len(W4_pre)):
    #     for j in range(len(W4_pre[0])):
    #         W4_pre[i,j] = A[ct]
    #         ct += 1
    # for i in range (len(W4_post)):
    #     for j in range(len(W4_post[0])):
    #         W4_post[i,j] = A[ct]
    #         ct += 1

    for i in range (len(W1[0])):
        for j in range(len(W1)):
            W1[j,i] = A[ct]
            ct += 1
    for i in range (len(W3_pre[0])):
        for j in range(len(W3_pre)):
            W3_pre[j,i] = A[ct]
            ct += 1
    for i in range (len(W3_post[0])):
        for j in range(len(W3_post)):
            W3_post[j,i] = A[ct]
            ct += 1
    for i in range (len(W4_pre[0])):
        for j in range(len(W4_pre)):
            W4_pre[j,i] = A[ct]
            ct += 1
    for i in range (len(W4_post[0])):
        for j in range(len(W4_post)):
            W4_post[j,i] = A[ct]
            ct += 1
    return(taus, W1, W3_pre, W3_post, W4_pre, W4_post)

def make_rule_str_TwvcMLP(taus, W1, W3_pre, W3_post, W4_pre, W4_post):
    rule_str = ""
    for i in taus:
        rule_str += "a"+str(np.exp(i))
    for i in W1.flatten():
        rule_str += "a"+str(i)
    for i in W3_pre.flatten():
        rule_str += "a"+str(i)
    for i in W3_post.flatten():
        rule_str += "a"+str(i)
    for i in W4_pre.flatten():
        rule_str += "a"+str(i)
    for i in W4_post.flatten():
        rule_str += "a"+str(i)
    return(rule_str+"a")

def make_rule_str_EEIE_TwvcPol(A):
    rule_EE_str, rule_IE_str = "", ""
    for i in range (4):
        rule_EE_str += "a"+str(np.exp(A[i]))
        rule_IE_str += "a"+str(np.exp(A[i+36]))
    for i in range(4,36):
        rule_EE_str += "a"+str(A[i])
        rule_IE_str += "a"+str(A[i+36])
    return(rule_EE_str+"a", rule_IE_str+"a")

def make_rule_str_4r6pPol(A):
    rule_string = " --ruleEE a" + "{}" + "a" + "{}"
    rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
    rule_string += " --ruleEI a" + "{}" + "a" + "{}"
    rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
    rule_string += " --ruleIE a" + "{}" + "a" + "{}"
    rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
    rule_string += " --ruleII a" + "{}" + "a" + "{}"
    rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
    return(rule_string.format(*A))

def compile_and_run_auryn_net(params):
    # Make call string for auryn sim
    params["id"] = 0
    if params["name"]=="sim_bg_TIF_IE_6pPol":
        cl_str = generate_call_auryn_bg_TIF_IE_6pPol(params)
    elif params["name"]=="sim_bg_IF_IE_6pPol":
        cl_str = generate_call_auryn_bg_IF_IE_6pPol(params)
    elif params["name"]=="sim_bg_TIF_EEIE_TwvcPol":
        cl_str = generate_call_auryn_bg_TIF_EEIE_TwvcPol(params)
    elif params["name"]=="sim_bg_TIF_IE_TwvcMLP":
        cl_str = generate_call_auryn_bg_TIF_IE_TwvcMLP(params)
    elif params["name"]=="sim_bg_IF_IE_TwvcMLP":
        cl_str = generate_call_auryn_bg_IF_IE_TwvcMLP(params)
    elif params["name"]=="sim_bg_AdEx_IE_6pPol":
        cl_str = generate_call_auryn_bg_AdEx_IE_6pPol(params)
    elif params["name"]=="sim_bg_IF_EEEIIEII_6pPol":
        cl_str = generate_call_auryn_bg_IF_EEEIIEII_6pPol(params)
    else:
        print('No call string function written for that innerloop name')
        raise NotImplementedError
    
    # compile and simulation code
    if os.path.exists(params["auryn_sim_dir"] + params["name"]): #don't launch a sim on outdated code if compilation fails
        os.remove(params["auryn_sim_dir"] + params["name"]) 
    compile_str = "cd " + params["auryn_sim_dir"]
    compile_str += " && make " + params["name"]
    output1 = subprocess.run(compile_str, shell=True, capture_output=True)
    start = time.time()
    output2 = subprocess.run(cl_str, shell=True, capture_output=True)
    exec_time = np.round(time.time() - start,3)
    return_str = "COMPILATION: \n \n Input: \n \n" + output1.args
    return_str += " \n \n Return: \n \n " + str(output1.stdout)[2:-1]
    return_str += " \n \n Potential errors: \n \n " + str(output1.stderr)[2:-1]
    return_str += " \n \nSIMULATION: \n \n Input: \n \n " + output2.args
    return_str += " \n \n Return: \n \n " + str(output2.stdout)[2:-1]
    return_str += " \n \n Potential errors: \n \n " + str(output2.stderr)[2:-1]
    return_str += " \n \n Simulation time: " + str(exec_time) + "s"
    return(return_str)

def empty_workdir(workdir_path): #TODO
    os.remove(filename_exc_spikes)

def get_compare_metric(function, file_paths, params):
    n_nets = len(file_paths)
    metric = np.full(n_nets, fill_value=np.nan)
    ct = 0
    for file_path in file_paths:
        metric[ct] = function(file_path, **params)
        ct += 1
    return(metric)

def get_compare_params_metric(function, file_paths, params_list):
    n_nets = len(file_paths)
    n_par = len(params_list)
    metric = np.full((n_nets, n_par), fill_value=np.nan)
    ct_p = 0
    for params in params_list:
        ct_n = 0
        for file_path in file_paths:
            metric[ct_n, ct_p] = function(file_path, **params)
            ct_n += 1
        ct_p += 1
    return(metric)