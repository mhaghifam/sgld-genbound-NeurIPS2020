import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from mnist_cnn_bounds import IncohBoundSimulator, GradBoundSimulator, CmiBoundSimulator


def main():
    num_iteration = 700
    num_runs = 100
    num_traindp = 20000
    sq_gen_inoch = np.zeros((num_runs, num_iteration))
    sq_gen_grad = np.zeros((num_runs, num_iteration))
    sq_gen_cmi = np.zeros((num_runs, num_iteration))

    for rep in range(num_runs):
        _, _, _, _, _, sq_gen_cmi_sample = CmiBoundSimulator().train()
        sq_gen_cmi[rep, :] = sq_gen_cmi_sample
        _, _, _, sq_gen_incoh_sample = IncohBoundSimulator().train()
        sq_gen_inoch[rep, :] = sq_gen_incoh_sample
        _, _, _, sq_gen_grad_sample = GradBoundSimulator().train()
        sq_gen_grad[rep, :] = sq_gen_grad_sample

    gen_incoh =  1/(2*np.sqrt(2))*np.mean(np.sqrt(sq_gen_inoch), axis=0)
    avg_gen_incoh = np.mean(gen_incoh, axis=0)
    std_gen_incoh = np.std(gen_incoh,axis=0)/np.sqrt(len(gen_incoh))
    gen_grad =np.sqrt(2) / num_traindp * np.sqrt(sq_gen_grad)
    avg_gen_grad = np.sqrt(2) / num_traindp * np.sqrt(np.mean(sq_gen_grad, axis=0))
    std_gen_grad = np.std(gen_grad,axis=0)/np.sqrt(len(gen_grad))
    gen_cmi = 1/(np.sqrt(2)*num_traindp)*np.sqrt(sq_gen_cmi)
    avg_gen_cmi = np.mean(gen_cmi, axis=0)
    std_gen_cmi = np.std(gen_cmi,axis=0)/np.sqrt(len(gen_cmi))

    fig = plt.figure()
    epochs = np.linspace(start=1, stop=num_iteration - 1, num=30, dtype=np.int32)
    plt.errorbar(epochs, avg_gen_incoh[epochs], yerr=std_gen_incoh[epochs], label='Negrea et al. (2019)',
                 color='green', linewidth=1)
    plt.errorbar(epochs, avg_gen_grad[epochs], yerr=std_gen_grad[epochs], label='Li, Luo, and Qiao (2020)',
                 color='red', linewidth=1)
    plt.errorbar(epochs, avg_gen_cmi[epochs], yerr=std_gen_cmi[epochs], label='CMI bound (ours)',
                 color='blue', linewidth=1)
    plt.xlabel('Training Iteration', fontsize=13)
    plt.ylabel('Expected generalization error', fontsize=20)
    plt.legend(loc='upper left', prop={'size': 20})
    plt.title('Expected generalization error', fontsize=30)
    plt.xticks(np.arange(0, num_iteration + 1, step=100))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=15)
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    main()