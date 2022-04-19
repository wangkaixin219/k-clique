from brain import *
from env import *

dim = 100


def train(graph, args):
    brain = Brain(state_dim=dim, action_dim=dim, hidden_dim=32, batch_size=128, gamma=0.99, K_epochs=80, eps_clip=0.2)
    env = Env(graph, args.k, dim=dim)
    
    epoch, n_epoch = 1, 10000
    best = 0
    prob = 1
    disc = 0.99

    while epoch <= n_epoch:
        state = env.reset()

        rewards = 0
        while not env.done:
            action = brain.select_action(state, prob=prob)
            reward, state, done = env.step(action)
            brain.buffer.rewards.append(reward)
            brain.buffer.done.append(done)
            rewards += reward

        calls = env.calculate_calls()
        brain.buffer.rewards[-1] += calls
        if calls > best:
            best = calls
            brain.save('best.pt'.format(epoch))
            print("Save current best model at epoch {}".format(epoch))

        print("******************** Epoch [{}/{}] ********************".format(epoch, n_epoch))
        print("Rewards = {}, Calls = {}".format(rewards, calls))
        brain.update()
        print("*******************************************************\n")

        if epoch % 10 == 0:
            # prob *= disc
            print("Explore prob: {}".format(prob))
            validate(graph, args)

        epoch += 1


def validate(graph, args):
    brain = Brain(state_dim=dim, action_dim=dim, hidden_dim=32, batch_size=128, gamma=0.99, K_epochs=80, eps_clip=0.2)
    env = Env(graph, args.k, dim=dim)
    
    brain.load('best.pt')
    state = env.reset()

    rewards = 0
    while not env.done:
        action = brain.select_action(state, prob=0)
        reward, state, done = env.step(action)
        rewards += reward

    calls = env.calculate_calls()

    print("*************** Validate ***************")
    print("Rewards = {}, Calls = {}".format(rewards, calls))
    print("****************************************\n")


