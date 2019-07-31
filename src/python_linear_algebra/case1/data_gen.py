import numpy as np

# init
init = np.array([0.85, 0, 0, 0.15])

# BMES
transition = np.array([[0, 0.2, 0.8, 0],
                       [0, 0.1, 0.9, 0],
                       [0.9, 0, 0, 0.1],
                       [0.9, 0, 0, 0.1]])

# ABCDEFG
np.random.seed(seed=9999)
condition = np.random.random((4, 7))
condition = condition/ np.repeat(np.sum(condition, axis=1, keepdims=True), 7, axis=1)

init_steps = np.dot(np.tri(4), init.transpose())
print(init_steps)
transition_steps = np.dot(np.tri(4), transition.transpose()).transpose()
print(transition_steps)
condition_steps = np.dot(np.tri(7), condition.transpose()).transpose()
print(condition_steps)

def discrete_sample(steps):
    size = len(steps)
    return np.where(np.random.random() < steps)[0][0]

def gen(n, length):
    obs = []
    states = []

    for i in range(n):
        one_seq = []
        sta_seq = []
        state = discrete_sample(init_steps)
        observation = discrete_sample(condition_steps[state])
        one_seq.append(observation)
        sta_seq.append(state)
    
        for j in range(length):
            state = discrete_sample(transition_steps[state])
            one_seq.append(discrete_sample(condition_steps[state]))
            sta_seq.append(state)

        obs.append(one_seq)
        states.append(sta_seq)
    return obs, states

obs, states = gen(3, 10)
for i in range(3):
    print(i)
    print(obs[i])
    print(states[i])

np.savetxt('init.dat', init)
np.savetxt('transition.dat', transition)
np.savetxt('condition.dat', condition)
np.savetxt('data.dat', np.array(obs))
np.savetxt('states.dat', np.array(states))
