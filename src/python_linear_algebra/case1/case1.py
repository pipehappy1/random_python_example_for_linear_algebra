import numpy as np

def dp(init, transition, condition, data):
    # init is the states' prior, like
    # 
    # transition is the transition matrix,
    #     the probility from state i to state j
    #     can be found at ith row, jth column.
    #
    # condition is the conditional distribution
    #     of the observation given the state.
    #     The probility of overvation m given the state j
    #     can be found at jth row, mth column.
    #
    # data is a one dimensional array.
    #
    # output is the most possible sequence of states
    #     given the input sequence.
    #
    if len(init.shape) != 1:
        print("Error: expect init to be a one dimensional array.")
        return
    
    N = len(init)
    init = np.ma.log(init)
    init = init.filled(-1e9)

    if len(transition.shape) != 2:
        print("Error: expect transition to be a two dimensional array.")
        return
    if not np.all(np.equal(transition.shape, N)):
        print("Error: expect transition to be a square array.")
        return

    transition = np.ma.log(transition)
    transition = transition.filled(-1e9)

    if len(condition.shape) != 2:
        print("Error: expect condition to be a two dimensional array.")
        return
    if condition.shape[0] != N:
        print("Error: expect condition and transition has the same size of states.")
        return

    condition = np.ma.log(condition)
    condition = condition.filled(-1e9)

    if len(data.shape) != 1:
        print("Error: expect data to be a one dimensional array.")
        return
    if np.max(data) > (condition.shape[1] - 1):
        print("Error: expect elements of data can be found in condition.")
        return

    best_pre = []
    best = None

    for i in range(len(data)):
        if i == 0:
            best = init + transition[:, int(data[i])]
        else:
            all_best = transition + np.repeat(condition[:, int(data[i])][None].T, N, axis=1).T + np.repeat(best[:][None].T, N, axis=1)
            best = np.max(all_best, axis=0)
            best_pre_state = np.argmax(all_best, axis=0)
            best_pre.append(best_pre_state)

    best_state = []
    last_best = None
    for i in range(len(data)):
        if i == 0:
            best_state.append(np.argmax(best))
            last_best = best_pre[len(data) - i - 2][np.argmax(best)]
        else:
            best_state.append(last_best)
            last_best = best_pre[len(data) - i - 2][last_best]

    best_state.reverse()
    
    return best_state
            

if __name__ == "__main__":
    # if *.dat cannot be found. run data_gen.py first.
    init = np.loadtxt('init.dat')
    transition = np.loadtxt('transition.dat')
    condition = np.loadtxt('condition.dat')

    data = np.loadtxt('data.dat')

    print("The observed sequence is: ", data[0])
    possible_states = dp(init, transition, condition, data[0])
    print("The most possible sequence of state is: ", possible_states)
