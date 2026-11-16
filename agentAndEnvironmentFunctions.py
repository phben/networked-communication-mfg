from jax import vmap
from multiprocessing import Pool
import math
import numpy as np
import jax.numpy as jnp
import scipy.optimize as optimize
from jax import random
from jax import jit
from jax.lax import fori_loop
from functools import partial
from jax import config

config.update("jax_enable_x64", True)


@jit
def choose_action(subkey, stateT, policy, ACTIONS):
    action = random.choice(subkey, ACTIONS, p=policy[stateT])
    return action


@jit
def step(
    stateT,
    gamma,
    timeWithinK,
    muHatT,
    policy,
    ACTIONS,
    key,
    gameMode,
    numberAgents,
    GRIDDIMENSION,
    targetPositions,
    LAMBDA,
    regDiscReturn,
):
    actionT = choose_action(key, stateT, policy, ACTIONS)
    rewardT = reward_function(
        gameMode, numberAgents, stateT, actionT, muHatT, GRIDDIMENSION, targetPositions
    )

    # thisReturn = (gamma ** timeWithinK) * (rewardT + entropy_regularisation(LAMBDA, policy[stateT])) # removing entropy regularisation is equivalent to setting lambda to 0
    thisReturn = (gamma**timeWithinK) * (rewardT)
    regDiscReturn = regDiscReturn + thisReturn

    stateTPlus1 = transition_function(stateT, actionT, muHatT, GRIDDIMENSION)

    return actionT, rewardT, regDiscReturn, stateTPlus1


@jit
def step_all_agents(
    stateTs,
    policies,
    key,
    gamma,
    timeWithinK,
    muHatT,
    ACTIONS,
    actionTs,
    rewardTs,
    regDiscReturns,
    stateTPlus1s,
    gameMode,
    GRIDDIMENSION,
    targetPositions,
    LAMBDA,
):
    numberAgents = jnp.shape(stateTs)[0]
    subkeys = random.split(key, numberAgents + 1)
    key = subkeys[0]
    actionTs, rewardTs, regDiscReturns, stateTPlus1s = vmap(
        step,
        (0, None, None, None, 0, None, 0, None, None, None, None, None, 0),
        (0, 0, 0, 0),
    )(
        stateTs,
        gamma,
        timeWithinK,
        muHatT,
        policies,
        ACTIONS,
        subkeys[1:],
        gameMode,
        numberAgents,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
        regDiscReturns,
    )

    return (
        actionTs,
        jnp.asarray(rewardTs, dtype=jnp.float64),
        jnp.asarray(regDiscReturns, dtype=jnp.float64),
        stateTPlus1s,
        key,
    )


@jit
def store_transitions(
    stateTMinus2s,
    actionTMinus2s,
    rewardTMinus2s,
    stateTMinus1s,
    actionTMinus1s,
    rewardTMinus1s,
    stateTs,
    actionTs,
    rewardTs,
    visitCounts,
    stateTPlus1s,
):
    stateTMinus2s = stateTMinus1s
    actionTMinus2s = actionTMinus1s
    rewardTMinus2s = rewardTMinus1s

    stateTMinus1s = stateTs
    actionTMinus1s = actionTs
    rewardTMinus1s = rewardTs

    stateTs = stateTPlus1s

    actionTs, rewardTs, stateTPlus1s = reset_for_store(actionTs, rewardTs, stateTPlus1s)

    visitCounts = vmap(update_visitCounts, (0, 0), 0)(visitCounts, stateTs)

    return (
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        stateTs,
        actionTs,
        rewardTs,
        visitCounts,
        stateTPlus1s,
    )


@jit
def update_visitCounts(agent_visitCount, stateT):
    return agent_visitCount.at[stateT].add(1)


@jit
def reset_for_store(actionTs, rewardTs, stateTPlus1s):
    actionTs = jnp.empty((jnp.shape(actionTs)), jnp.int8)
    rewardTs = jnp.empty((jnp.shape(rewardTs)), jnp.float64)
    stateTPlus1s = jnp.empty((jnp.shape(stateTPlus1s)), jnp.int64)

    return actionTs, rewardTs, stateTPlus1s


@jit
def add_to_all_batches(
    numberAgents,
    stateTMinus2s,
    actionTMinus2s,
    rewardTMinus2s,
    stateTMinus1s,
    actionTMinus1s,
    batches,
    mpg,
):
    batches = vmap(add_to_one_batch, (0, 0, 0, 0, 0, 0, None), 0)(
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        batches,
        mpg,
    )
    return batches


@jit
def add_to_one_batch(
    stateTMinus2, actionTMinus2, rewardTMinus2, stateTMinus1, actionTMinus1, batch, mpg
):
    transition = jnp.array(
        [stateTMinus2, actionTMinus2, rewardTMinus2, stateTMinus1, actionTMinus1]
    )
    return batch.at[mpg].set(transition)


@jit
def batch_learn_for_all(
    numberAgents,
    batches,
    learningIterationsL,
    learningRateBeta,
    allQvalues,
    policies,
    GAMMA,
    key,
    LAMBDA,
    onlines,
):
    subkeys = random.split(key, jnp.shape(policies)[0] + 1)
    key = subkeys[0]
    allQvalues = vmap(batch_learn_for_one, (0, None, None, 0, 0, None, 0, None, 0), 0)(
        batches,
        learningIterationsL,
        learningRateBeta,
        allQvalues,
        policies,
        GAMMA,
        subkeys[1:],
        LAMBDA,
        onlines,
    )

    batches = jnp.empty(jnp.shape(batches))
    return allQvalues, batches, key


@jit
def batch_learn_for_one(
    batch,
    learningIterationsL,
    learningRateBeta,
    Qvalues,
    policy,
    GAMMA,
    subkey,
    LAMBDA,
    online,
):
    newQs = jnp.select(
        [online],
        [
            batch_learn(
                batch,
                learningIterationsL,
                learningRateBeta,
                Qvalues,
                policy,
                GAMMA,
                subkey,
                LAMBDA,
            )[0]
        ],
        Qvalues,
    )
    return newQs


@jit
def get_a_shuffle(key, batch):
    key, subkey = random.split(key)
    batch = random.permutation(subkey, batch, independent=False)

    return key, batch


@partial(jit, static_argnames=["learningIterationsL"])
def batch_learn(
    batch, learningIterationsL, learningRateBeta, Qvalues, policy, GAMMA, key, LAMBDA
):
    args = batch, key, Qvalues, learningRateBeta, policy, GAMMA, LAMBDA
    args = fori_loop(0, learningIterationsL, use_batch, args)
    batch, key, Qvalues, learningRateBeta, policy, GAMMA, LAMBDA = args
    return Qvalues, key


@jit
def use_batch(learningIteration, args):
    batch, key, Qvalues, learningRateBeta, policy, GAMMA, LAMBDA = args

    args = learningRateBeta, Qvalues, policy, GAMMA, LAMBDA, batch
    args = fori_loop(0, jnp.shape(batch)[0], TD_update, args)

    learningRateBeta, Qvalues, policy, GAMMA, LAMBDA, batch = args

    return batch, key, Qvalues, learningRateBeta, policy, GAMMA, LAMBDA


@jit
def TD_update(batch_index, args):
    learningRate, Qvalues, policy, GAMMA, LAMBDA, batch = args
    transition = batch[batch_index]

    state = transition[0]
    action = transition[1]
    reward = transition[2]
    statePrime = transition[3]
    actionPrime = transition[4]

    intermediate = (
        Qvalues[state.astype(int), action.astype(int)]
        - reward
        - (GAMMA * Qvalues[statePrime.astype(int), actionPrime.astype(int)])
    )
    newVal = (
        Qvalues[state.astype(int), action.astype(int)] - learningRate * intermediate
    )

    Qvalues = Qvalues.at[state.astype(int), action.astype(int)].set(newVal)
    args = learningRate, Qvalues, policy, GAMMA, LAMBDA, batch
    return args


def update_all_policies(lowerBound, policies, allQvalues, LAMBDA, eta, onlines):
    with Pool() as pool:
        args = [
            (
                lowerBound,
                policies[agent],
                allQvalues[agent],
                LAMBDA,
                eta,
                onlines[agent],
            )
            for agent in jnp.arange(jnp.shape(policies)[0])
        ]
        policies = jnp.array(pool.starmap(PMA_update, args))

    return policies


def update_one_policy(agent, args):
    lowerBound, policies, allQvalues, LAMBDA, eta = args

    newPol = PMA_update(lowerBound, policies[agent], allQvalues[agent], LAMBDA, eta)
    policies = policies.at[agent].set(newPol)
    return lowerBound, policies, allQvalues, LAMBDA, eta


def PMA_update(lowerBound, policy, Qvalues, LAMBDA, eta, online):
    if online:
        # setConstraint = optimize.NonlinearConstraint(entropy_regularisation, lowerBound, jnp.inf) # removing the bound as discussed in Sec. 7.4.1
        distributionConstraint = optimize.LinearConstraint(
            np.ones(jnp.shape(policy)[1]), lb=1, ub=1
        )  # to make sure probabilities of all actions sum to 1
        bounds = [
            (5e-324, 1) for action in range(jnp.shape(policy)[1])
        ]  # 5e-324 so probability of each action has to be more than 0

        for thisState in jnp.arange(jnp.shape(policy)[0]):
            newStateDistribution = optimize.minimize(
                optimisation_function,
                x0=np.array(policy[thisState]),
                args=(thisState, Qvalues, policy, eta, LAMBDA, lowerBound),
                constraints=[distributionConstraint],
                bounds=bounds,
            ).x

            policy = policy.at[thisState].set(newStateDistribution)

    return policy


def optimise_for_state(
    thisState, Qvalues, policy, eta, LAMBDA, lowerBound, distributionConstraint, bounds
):
    return optimize.minimize(
        optimisation_function,
        x0=np.array(policy[thisState]),
        args=(thisState, Qvalues, policy, eta, LAMBDA, lowerBound),
        constraints=[distributionConstraint],
        bounds=bounds,
    ).x


@jit
def maximisation_function(
    actionDistribution, state, Qvalues, policy, eta, LAMBDA, lowerBound
):
    product = jnp.dot(actionDistribution, Qvalues[state])
    # entropy = entropy_regularisation(LAMBDA, actionDistribution) # removing entropy regularisation is equivalent to setting lambda to 0
    norm = jnp.linalg.norm(jnp.subtract(actionDistribution, policy[state]))

    # result = product + entropy - ((norm ** 2) / (2 * eta)) # removing entropy regularisation is equivalent to setting lambda to 0
    result = product - ((norm**2) / (2 * eta))
    return result


# to make it a minimisation problem
@jit
def optimisation_function(
    actionDistribution, thisState, Qvalues, policy, eta, LAMBDA, lowerBound
):
    return -1 * maximisation_function(
        actionDistribution, thisState, Qvalues, policy, eta, LAMBDA, lowerBound
    )


# returns the number of grid positions
@jit
def give_states(GRIDDIMENSION):
    return GRIDDIMENSION**2


@jit
def action0row(row):
    return row


@jit
def action1row(row):
    new_row = row - 1
    new_row = jnp.select([new_row < 0], [row], new_row)
    return new_row


@jit
def action2row(row):
    new_row = row
    return new_row


@jit
def action3row(row, GRIDDIMENSION):
    new_row = row + 1
    new_row = jnp.select([new_row > (GRIDDIMENSION - 1)], [row], new_row)
    return new_row


@jit
def action4row(row):
    new_row = row
    return new_row


@jit
def action0col(col):
    return col


@jit
def action1col(col):
    new_col = col
    return new_col


@jit
def action2col(col, GRIDDIMENSION):
    new_col = col + 1
    new_col = jnp.select([new_col > (GRIDDIMENSION - 1)], [col], new_col)
    return new_col


@jit
def action3col(col):
    new_col = col
    return new_col


@jit
def action4col(col):
    new_col = col - 1
    new_col = jnp.select([new_col < 0], [col], new_col)
    return new_col


@jit
def transition_function(state, action, muHat, GRIDDIMENSION):
    position = getPosFromState(state, GRIDDIMENSION)
    row = position[0]
    col = position[1]
    new_row = None
    new_col = None

    # 1 == up, 2 == right, 3 == down, 4 == left, 0 == remain
    # make move but stay still if out of range
    new_row = jnp.select(
        [action == 0, action == 1, action == 2, action == 3, action == 4],
        [
            action0row(row),
            action1row(row),
            action2row(row),
            action3row(row, GRIDDIMENSION),
            action4row(row),
        ],
    )
    new_col = jnp.select(
        [action == 0, action == 1, action == 2, action == 3, action == 4],
        [
            action0col(col),
            action1col(col),
            action2col(col, GRIDDIMENSION),
            action3col(col),
            action4col(col),
        ],
    )

    newState = getStateFromPos(new_row, new_col, GRIDDIMENSION)
    return newState


@jit
def getStateFromPos(row, col, GRIDDIMENSION):
    state = row * GRIDDIMENSION + col
    return state


@jit
def getPosFromState(state, GRIDDIMENSION):
    row = state // GRIDDIMENSION
    col = state % GRIDDIMENSION

    position = jnp.array([row, col])
    return position


@jit
def cluster_reward(muHat, state, numberAgents):
    maxClusterReward = jnp.log(1)
    minClusterReward = jnp.log(1 / numberAgents)

    reward = jnp.log(muHat[state])

    normalisedReward = (reward - minClusterReward) / (
        maxClusterReward - minClusterReward
    )
    return normalisedReward


@jit
def agree_reward(numberAgents, state, action, muHat, GRIDDIMENSION, targetPositions):
    minClusterReward = -1
    maxClusterReward = 1

    agentPosition = getPosFromState(state, GRIDDIMENSION)
    reward = -1

    initArgs = reward, targetPositions, agentPosition, numberAgents, state, muHat
    args = fori_loop(
        0, jnp.shape(targetPositions)[0], agree_reward_helper_main, initArgs
    )
    reward, targetPositions, agentPosition, numberAgents, state, muHat = args

    normalisedReward = (reward - minClusterReward) / (
        maxClusterReward - minClusterReward
    )
    return normalisedReward


@jit
def agree_reward_helper_main(target, args):
    reward, targetPositions, agentPosition, numberAgents, state, muHat = args
    dist = abs(agentPosition[0] - targetPositions[target, 0]) + abs(
        agentPosition[1] - targetPositions[target, 1]
    )
    reward = jnp.select(
        [dist == 0], [agree_reward_helper(muHat, state, numberAgents, reward)], reward
    )
    return reward, targetPositions, agentPosition, numberAgents, state, muHat


@jit
def agree_reward_helper(muHat, state, numberAgents, reward):
    return jnp.select([muHat[state] > 1 / numberAgents], [muHat[state]], reward)


@jit
def reward_function(
    gameMode, numberAgents, state, action, muHat, GRIDDIMENSION, targetPositions
):
    return jnp.select(
        [gameMode == 0, gameMode == 1],
        [
            cluster_reward(muHat, state, numberAgents),
            agree_reward(
                numberAgents, state, action, muHat, GRIDDIMENSION, targetPositions
            ),
        ],
    )


@jit
def initialise_state(key, NUMSTATES):
    return random.randint(key, [1], 0, NUMSTATES)


@jit
def shareAndAdoptPolicies(
    stateTs,
    policies,
    sigmas,
    communicationRadius,
    soft,
    key,
    GRIDDIMENSION,
    temperature,
):
    subkeys = random.split(key, jnp.shape(stateTs)[0] + 1)
    key = subkeys[0]
    newPolicies, newSigmas = vmap(
        receiveForAgentI, (None, None, 0, None, None, 0, None, None, None), (0, 0)
    )(
        policies,
        sigmas,
        jnp.arange(jnp.shape(stateTs)[0]),
        stateTs,
        communicationRadius,
        subkeys[1:],
        soft,
        GRIDDIMENSION,
        temperature,
    )

    return newPolicies, newSigmas, key


@jit
def receiveForAgentI(
    policies,
    sigmas,
    agentI,
    stateTs,
    communicationRadius,
    subkey,
    soft,
    GRIDDIMENSION,
    temperature,
):
    agentIReceivedPolicies, agentIReceivedSigmas = collectAllCommunicated(
        stateTs, agentI, communicationRadius, sigmas, policies, GRIDDIMENSION
    )

    newPolicy = policies[agentI]
    newSigma = -1000000

    newPolicy = jnp.select(
        [jnp.size(agentIReceivedSigmas) != 0 and soft == False],  # noqa: E712
        [adoptMax_pol(agentIReceivedSigmas, agentIReceivedPolicies)],
        newPolicy,
    )
    newSigma = jnp.select(
        [jnp.size(agentIReceivedSigmas) != 0 and soft == False],  # noqa: E712
        [adoptMax_sig(agentIReceivedSigmas)],
        newSigma,
    )

    softmaxIndex = jnp.select(
        [jnp.shape(agentIReceivedSigmas)[0] != 0 and soft == True],  # noqa: E712
        [get_softmaxIndex(agentIReceivedSigmas, subkey, temperature)],
        -1,
    )

    newPolicy = jnp.select(
        [softmaxIndex != -1], [agentIReceivedPolicies[softmaxIndex]], newPolicy
    )
    newSigma = jnp.select(
        [softmaxIndex != -1], [agentIReceivedSigmas[softmaxIndex]], newSigma
    )

    return newPolicy, newSigma


@jit
def get_softmaxIndex(agentIReceivedSigmas, subkey, temperature):
    probabilities = softmax(agentIReceivedSigmas, temperature)
    indices = jnp.arange(jnp.size(agentIReceivedSigmas))
    softmaxIndex = random.choice(subkey, indices, p=probabilities)
    return softmaxIndex


@jit
def adoptMax_pol(agentIReceivedSigmas, agentIReceivedPolicies):
    index = jnp.argmax(agentIReceivedSigmas)
    newPolicy = agentIReceivedPolicies[index]
    return newPolicy


@jit
def adoptMax_sig(agentIReceivedSigmas):
    index = jnp.argmax(agentIReceivedSigmas)
    newSigma = agentIReceivedSigmas[index]
    return newSigma


@jit
def collectAllCommunicated(
    stateTs, agentI, communicationRadius, sigmas, policies, GRIDDIMENSION
):
    agentIPosition = getPosFromState(stateTs[agentI], GRIDDIMENSION)
    agentIReceivedPolicies, agentIReceivedSigmas = vmap(
        collectACommunicated, (0, 0, None, None, 0, 0, None, None), (0, 0)
    )(
        jnp.arange(jnp.shape(stateTs)[0]),
        stateTs,
        agentI,
        communicationRadius,
        sigmas,
        policies,
        agentIPosition,
        GRIDDIMENSION,
    )

    return agentIReceivedPolicies, agentIReceivedSigmas


@jit
def collectACommunicated(
    agentJ,
    stateTJ,
    agentI,
    communicationRadius,
    sigmaJ,
    policyJ,
    agentIPosition,
    GRIDDIMENSION,
):
    agentJPosition = getPosFromState(stateTJ, GRIDDIMENSION)
    distance = jnp.linalg.norm(jnp.subtract(agentIPosition, agentJPosition))

    agentIReceivedPolicy = collectACommunicated_helper_pol(
        communicationRadius, sigmaJ, policyJ, distance
    )
    agentIReceivedSigma = collectACommunicated_helper_sig(
        communicationRadius, sigmaJ, distance
    )

    return agentIReceivedPolicy, agentIReceivedSigma


@jit
def collectACommunicated_helper_pol(communicationRadius, sigmaJ, policyJ, distance):
    agentIReceivedPolicy = jnp.select(
        [distance <= communicationRadius],
        [collectACommunicated_helper_two_pol(sigmaJ, policyJ)],
        -1 * jnp.inf,
    )
    return agentIReceivedPolicy


@jit
def collectACommunicated_helper_sig(communicationRadius, sigmaJ, distance):
    agentIReceivedSigma = jnp.select(
        [distance <= communicationRadius],
        [collectACommunicated_helper_two_sig(sigmaJ)],
        -1 * jnp.inf,
    )
    return agentIReceivedSigma


@jit
def collectACommunicated_helper_two_pol(sigmaJ, policyJ):
    agentIReceivedPolicy = jnp.select([sigmaJ != -1000000], [policyJ], -1 * jnp.inf)
    return agentIReceivedPolicy


@jit
def collectACommunicated_helper_two_sig(sigmaJ):
    agentIReceivedSigma = jnp.select([sigmaJ != -1000000], [sigmaJ], -1 * jnp.inf)
    return agentIReceivedSigma


@jit
def softmax(vector, temperature):
    e = jnp.exp(vector / temperature)
    result = e / jnp.sum(e, dtype=jnp.float64)

    return result


@partial(jit, static_argnames=["numberAgents", "NUMSTATES", "NUMACTIONS"])
def initialise_policies(numberAgents, NUMSTATES, NUMACTIONS):
    return jnp.full((numberAgents, NUMSTATES, NUMACTIONS), 1 / NUMACTIONS)


@partial(jit, static_argnames=["numberAgents", "NUMSTATES", "NUMACTIONS"])
def resetQs(numberAgents, NUMSTATES, NUMACTIONS, qmax):
    return jnp.full((numberAgents, NUMSTATES, NUMACTIONS), qmax)


@jit
def entropy_regularisation(LAMBDA, actionDistribution):
    runningTotals = vmap(add_to_runningTotal, 0, 0)(actionDistribution)
    total = jnp.sum(runningTotals)
    return -1 * LAMBDA * total


@jit
def add_to_runningTotal(action_prob):
    action_prob = jnp.select(
        [action_prob < 5e-324], [5e-324], action_prob
    )  # to handle if probabilities get too close to 0 (can't do log 0)
    return action_prob * jnp.log(action_prob)


@partial(jit, static_argnames=["numberAgents", "NUMSTATES"])
def reset_visitCounts(numberAgents, NUMSTATES):
    return jnp.full((numberAgents, NUMSTATES), 0)


@partial(jit, static_argnames=["numberAgents", "NUMSTATES", "NUMACTIONS"])
def reset_agents(numberAgents, NUMSTATES, NUMACTIONS, qmax):
    allQvalues = resetQs(numberAgents, NUMSTATES, NUMACTIONS, qmax)
    visitCounts = reset_visitCounts(numberAgents, NUMSTATES)
    regDiscReturns = jnp.full(numberAgents, 0, jnp.float64)
    sigmas = jnp.full(numberAgents, -1000000, jnp.float64)

    return allQvalues, visitCounts, regDiscReturns, sigmas


def print_in_grid_shape(list, NUMSTATES, filename=None):
    row = []
    for i, action in enumerate(list):
        row.append(action)
        if i % (math.sqrt(NUMSTATES)) == (math.sqrt(NUMSTATES) - 1):
            if filename:
                print(row, file=filename)
            else:
                print(row)
            row.clear()


def print_policy(policy, NUMSTATES, file=None):
    maxPolicy = vmap(get_argmax, (0), 0)(policy)
    maxPolicy = np.array(maxPolicy)

    print_in_grid_shape(maxPolicy, NUMSTATES, file)


def print_Qfunction(Qfunction, NUMSTATES, file=None):
    maxQfunction = vmap(get_argmax, (0), 0)(Qfunction)
    maxQfunction = np.array(maxQfunction)

    print_in_grid_shape(maxQfunction, NUMSTATES, file)


@jit
def get_argmax(state):
    return jnp.argmax(state)


def print_visitCount(visitCount, NUMSTATES):
    print_in_grid_shape(visitCount, NUMSTATES)
