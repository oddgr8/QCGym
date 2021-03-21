import matplotlib.pyplot as plt
import numpy as np

def pulse_plot(action_seq, fidelity):
    action_seq = np.array(action_seq)
    x = np.arange(0, len(action_seq[:,0,0]))
    fig, axs = plt.subplots(2, 2)
    leg = ["omega1", "omega2", "phi1", "phi2"] 
    for i in range(2):
        for j in range(2):
            # plt.plot(x, self.action_seq_best[:,i,j], linewidth=1)
            # plt.scatter(x, self.action_seq_best[:,i,j], s=2)
            axs[i,j].step(x, action_seq[:,i,j])
            axs[i,j].set_title(leg[i*2+j])
            axs[i,j].grid()
    
    plt.xlabel("time")
    plt.ylabel("sequence")
    plt.legend()
    plt.savefig('plt/pulse_seq_best'+str(round(fidelity,3))+'.png')
    plt.close()

def pulse_plot_bb(action_seq, fidelity, N):
    action_seq = np.array(action_seq)
    action_seq = action_seq/(N-1)

    f = open('pulse_seq/pulse_seq_best'+str(round(fidelity,1))+'.txt', 'a+')
    f.write(str(list(action_seq))+'\n')
    f.write(str(fidelity)+ " " + str(N)+'\n')
    f.close()
    x = np.arange(0, len(action_seq))
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(h_pad=2)
    leg = ["omega1", "omega2", "phi1", "phi2"] 
    colors = ['ro','go','bo','ro']
    for i in range(4):
        # plt.plot(x, self.action_seq_best[:,i,j], linewidth=1)
        axs[i%2,i//2].step(x, action_seq[:,i])
        axs[i%2,i//2].scatter(x, action_seq[:,i], s=3)
        axs[i%2,i//2].set_title(leg[i])
        axs[i%2,i//2].grid()
    
    plt.xlabel("time")
    plt.ylabel("sequence")
    plt.legend()
    plt.savefig('plt/pulse_seq_best'+str(round(fidelity,3))+'.png')
    plt.close()