# ruff: noqa: F405
# ruff: noqa: F403
import numpy as np
import jax.numpy as jnp
from jax import random
import math
from jax import jit
from agentAndEnvironmentFunctions import *
from jax import debug
from jax.lax import fori_loop
from jax import config

config.update("jax_enable_x64", True)


def run_algorithm(
    trial,
    gameMode,
    algoType,
    numberAgents,
    NUMACTIONS,
    GAMMA,
    LAMBDA,
    maxK,
    maxMpg,
    maxMtd,
    GRIDDIMENSION,
    eta,
    lh,
    testExploitation,
    testRobustness,
    communicationFrac,
    learningIterationsL,
    learningRateBeta,
    maxSharingIterationsC,
    oneTimeIncrease,
    soft,
    temperature,
    evalIterations,
):
    if gameMode == 0:
        game = "cluster"
    elif gameMode == 1:
        game = "agree on a single target"

    if algoType == "centralised":
        algoCode = 0
    elif algoType == "independent":
        algoCode = 1
    elif algoType == "networkedEvalPol":
        algoCode = 2
    else:
        algoCode = None

    seed = trial + algoCode * 100 + soft * 10000 + 1000
    if communicationFrac is not None:
        seed += int(communicationFrac * 10)
    if temperature is not None:
        seed += temperature * 1000000000
    key = random.PRNGKey(int(seed))

    label = (
        game
        + "; dimension = "
        + str(GRIDDIMENSION)
        + "; agents = "
        + str(numberAgents)
        + "; lambda = "
        + str(LAMBDA)
        + "; eta = "
        + str(eta)
        + "; maxK = "
        + str(maxK)
        + "; maxMpg = "
        + str(maxMpg)
        + "; maxMtd = "
        + str(maxMtd)
        + "; "
        + algoType
    )
    if communicationFrac is not None:
        label += str(communicationFrac) + "; sharingIts = " + str(maxSharingIterationsC)
    if testRobustness is not None:
        label += "; " + testRobustness
    if oneTimeIncrease is not None:
        label += str(oneTimeIncrease)

    ACTIONS = jnp.array([0, 1, 2, 3, 4])

    NUMSTATES = GRIDDIMENSION**2
    NUMACTIONS = 5
    communicationRadius = (
        communicationFrac * math.sqrt(2 * (GRIDDIMENSION**2))
        if communicationFrac is not None
        else None
    )

    targetPosition1 = jnp.array([0, 0])
    targetPosition2 = jnp.array([GRIDDIMENSION - 1, 0])
    targetPosition3 = jnp.array([GRIDDIMENSION - 1, GRIDDIMENSION - 1])
    targetPosition4 = jnp.array([0, GRIDDIMENSION - 1])

    targetPositions = jnp.array(
        [targetPosition1, targetPosition2, targetPosition3, targetPosition4]
    )

    if testRobustness == "one_time_addition":
        numberSpareAgents = numberAgents - int((1 / 5) * numberAgents)
        numberAgents = numberAgents - numberSpareAgents

    # intitialising agents
    policies = initialise_policies(numberAgents, NUMSTATES, NUMACTIONS)
    hmax = entropy_regularisation(LAMBDA, policies[0, 0])
    qmax = (1 + hmax) / (1 - GAMMA)
    lowerBound = (
        hmax - lh
    )  # the bounds as we define in Sec. 3.1; we ignore this bound in practice

    allQvalues, visitCounts, regDiscReturns, sigmas = reset_agents(
        numberAgents, NUMSTATES, NUMACTIONS, qmax
    )

    stateTs = jnp.empty((numberAgents), jnp.int64)
    actionTs = jnp.empty((numberAgents), jnp.int8)
    rewardTs = jnp.empty((numberAgents), jnp.float64)

    stateTMinus1s = jnp.empty((numberAgents), jnp.int64)
    actionTMinus1s = jnp.empty((numberAgents), jnp.int64)
    rewardTMinus1s = jnp.empty((numberAgents), jnp.float64)

    stateTMinus2s = jnp.empty((numberAgents), jnp.int64)
    actionTMinus2s = jnp.empty((numberAgents), jnp.int64)
    rewardTMinus2s = jnp.empty((numberAgents), jnp.float64)

    stateTPlus1s = jnp.empty((numberAgents), jnp.int64)

    batchLength = maxMpg - 1
    batches = jnp.empty((numberAgents, batchLength, 5))

    for agent in range(numberAgents):
        key, subkey = random.split(key)
        stateT = initialise_state(subkey, NUMSTATES)
        stateTs = stateTs.at[agent].set(stateT[0])

    onlines = jnp.full((numberAgents), 1)
    if testRobustness == "continued_random_failures":
        failureProbability = 0.5

    def one_time_addition(
        key,
        numberSpareAgents,
        numberAgents,
        policies,
        allQvalues,
        visitCounts,
        regDiscReturns,
        sigmas,
        stateTs,
        actionTs,
        rewardTs,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTPlus1s,
        batches,
        maxMpg,
        onlines,
    ):
        spare_policies = initialise_policies(numberSpareAgents, NUMSTATES, NUMACTIONS)
        policies = jnp.concatenate((policies, spare_policies))

        spare_allQvalues, spare_visitCounts, spare_regDiscReturns, spare_sigmas = (
            reset_agents(numberSpareAgents, NUMSTATES, NUMACTIONS, qmax)
        )
        allQvalues = jnp.concatenate((allQvalues, spare_allQvalues))
        visitCounts = jnp.concatenate((visitCounts, spare_visitCounts))
        regDiscReturns = jnp.concatenate((regDiscReturns, spare_regDiscReturns))
        sigmas = jnp.concatenate((sigmas, spare_sigmas))

        spare_stateTs = jnp.empty((numberSpareAgents), jnp.int64)
        for agent in range(numberSpareAgents):
            key, subkey = random.split(key)
            stateT = initialise_state(subkey, NUMSTATES)
            spare_stateTs = spare_stateTs.at[agent].set(stateT[0])

        stateTs = jnp.concatenate((stateTs, spare_stateTs))
        spare_actionTs = jnp.empty((numberSpareAgents), jnp.int8)
        actionTs = jnp.concatenate((actionTs, spare_actionTs))
        spare_rewardTs = jnp.empty((numberSpareAgents), jnp.float64)
        rewardTs = jnp.concatenate((rewardTs, spare_rewardTs))

        spare_stateTMinus1s = jnp.empty((numberSpareAgents), jnp.int64)
        stateTMinus1s = jnp.concatenate((stateTMinus1s, spare_stateTMinus1s))
        spare_actionTMinus1s = jnp.empty((numberSpareAgents), jnp.int64)
        actionTMinus1s = jnp.concatenate((actionTMinus1s, spare_actionTMinus1s))
        spare_rewardTMinus1s = jnp.empty((numberSpareAgents), jnp.float64)
        rewardTMinus1s = jnp.concatenate((rewardTMinus1s, spare_rewardTMinus1s))

        spare_stateTMinus2s = jnp.empty((numberSpareAgents), jnp.int64)
        stateTMinus2s = jnp.concatenate((stateTMinus2s, spare_stateTMinus2s))
        spare_actionTMinus2s = jnp.empty((numberSpareAgents), jnp.int64)
        actionTMinus2s = jnp.concatenate((actionTMinus2s, spare_actionTMinus2s))
        spare_rewardTMinus2s = jnp.empty((numberSpareAgents), jnp.float64)
        rewardTMinus2s = jnp.concatenate((rewardTMinus2s, spare_rewardTMinus2s))

        spare_stateTPlus1s = jnp.empty((numberSpareAgents), jnp.int64)
        stateTPlus1s = jnp.concatenate((stateTPlus1s, spare_stateTPlus1s))

        batchLength = maxMpg - 1
        spare_batches = jnp.empty((numberSpareAgents, batchLength, 5))
        batches = jnp.concatenate((batches, spare_batches))

        spare_onlines = jnp.full((numberSpareAgents), 1)
        onlines = jnp.concatenate((onlines, spare_onlines))
        numberAgents = numberAgents + numberSpareAgents

        return (
            key,
            numberAgents,
            policies,
            allQvalues,
            visitCounts,
            regDiscReturns,
            sigmas,
            stateTs,
            actionTs,
            rewardTs,
            stateTMinus1s,
            actionTMinus1s,
            rewardTMinus1s,
            stateTMinus2s,
            actionTMinus2s,
            rewardTMinus2s,
            stateTPlus1s,
            batches,
            maxMpg,
            onlines,
        )

    averageReturnList = jnp.empty(maxK)
    policyNorms = jnp.empty(0)
    time = 0
    exploitabilityList = jnp.empty(0)
    exploitTestFrequency = 2

    for k in range(maxK):
        onlines = jnp.full((numberAgents), 1)
        # this section allows us to approximate exploitability as described in Sec. 7.2.1
        if testExploitation and (k % exploitTestFrequency == 0):
            # store current policy before deviation:
            storedPolicy = jnp.copy(policies[0])

            commonregDiscReturnList = jnp.empty(0)
            bestReturn = 0
            exploitIterations = 40  # 40 learning and finding best
            for exploitIteration in range(exploitIterations):
                timeWithinK = 0

                allQvalues, visitCounts, regDiscReturns, sigmas = reset_agents(
                    numberAgents, NUMSTATES, NUMACTIONS, qmax
                )

                singleLearner = True
                args = (
                    maxMtd,
                    stateTs,
                    policies,
                    key,
                    GAMMA,
                    timeWithinK,
                    ACTIONS,
                    actionTs,
                    rewardTs,
                    regDiscReturns,
                    stateTPlus1s,
                    time,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    visitCounts,
                    gameMode,
                    GRIDDIMENSION,
                    targetPositions,
                    LAMBDA,
                    singleLearner,
                    batches,
                    numberAgents,
                )
                args = fori_loop(0, maxMpg, mpg_step, args)
                (
                    maxMtd,
                    stateTs,
                    policies,
                    key,
                    GAMMA,
                    timeWithinK,
                    ACTIONS,
                    actionTs,
                    rewardTs,
                    regDiscReturns,
                    stateTPlus1s,
                    time,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    visitCounts,
                    gameMode,
                    GRIDDIMENSION,
                    targetPositions,
                    LAMBDA,
                    singleLearner,
                    batches,
                    IGNOREnumberAgents,
                ) = args

                print("k =", k)
                print("exploitability improvement iterations =", exploitIteration)

                totalvisitCount = jnp.sum(visitCounts, axis=0)
                visitTotal = jnp.sum(totalvisitCount)
                percentages = np.array(
                    jnp.round((totalvisitCount / visitTotal) * 100), dtype=np.int8
                )
                print("visit percentages:")
                print_in_grid_shape(percentages, NUMSTATES)

                agent = 0

                newQs, key = batch_learn(
                    batches[agent],
                    100,
                    learningRateBeta,
                    allQvalues[agent],
                    policies[agent],
                    GAMMA,
                    key,
                    LAMBDA,
                )

                allQvalues = allQvalues.at[agent].set(newQs)
                batches = jnp.empty((numberAgents, batchLength, 5))

                newPol = PMA_update(
                    lowerBound, policies[0], allQvalues[0], LAMBDA, eta, onlines[0]
                )
                policies = policies.at[0].set(newPol)

                commonregDiscReturnList = jnp.concatenate(
                    (commonregDiscReturnList, regDiscReturns[1:])
                )
                betterReturn = regDiscReturns[0]
                bestReturn = max(bestReturn, betterReturn)

            exploitability = bestReturn - (
                jnp.sum(commonregDiscReturnList) / jnp.size(commonregDiscReturnList)
            )
            exploitabilityList = jnp.append(exploitabilityList, exploitability)

            # return previous policy from before deviation:
            policies = policies.at[0].set(storedPolicy)

        # below is the real learning loop, ie not the one for approximating exploitability

        debug.print("k = " + str(k))
        timeWithinK = 0
        policyNorms = jnp.append(policyNorms, get_policy_norm(policies))
        allQvalues, visitCounts, regDiscReturns, sigmas = reset_agents(
            numberAgents, NUMSTATES, NUMACTIONS, qmax
        )

        @jit  # to set which agents randomly fail in any given iteration
        def get_failure(subkey, failureProbability):
            online = random.choice(
                subkey,
                jnp.asarray([1, 0]),
                p=jnp.asarray([1 - failureProbability, failureProbability]),
            )
            return online

        if testRobustness == "continued_random_failures":
            subkeys = random.split(key, jnp.shape(stateTs)[0] + 1)
            key = subkeys[0]
            onlines = vmap(get_failure, (0, None), 0)(subkeys[1:], failureProbability)

        singleLearner = True if algoType == "centralised" else False
        args = (
            maxMtd,
            stateTs,
            policies,
            key,
            GAMMA,
            timeWithinK,
            ACTIONS,
            actionTs,
            rewardTs,
            regDiscReturns,
            stateTPlus1s,
            time,
            stateTMinus2s,
            actionTMinus2s,
            rewardTMinus2s,
            stateTMinus1s,
            actionTMinus1s,
            rewardTMinus1s,
            visitCounts,
            gameMode,
            GRIDDIMENSION,
            targetPositions,
            LAMBDA,
            singleLearner,
            batches,
            numberAgents,
        )
        args = fori_loop(0, maxMpg, mpg_step, args)
        (
            maxMtd,
            stateTs,
            policies,
            key,
            GAMMA,
            timeWithinK,
            ACTIONS,
            actionTs,
            rewardTs,
            regDiscReturns,
            stateTPlus1s,
            time,
            stateTMinus2s,
            actionTMinus2s,
            rewardTMinus2s,
            stateTMinus1s,
            actionTMinus1s,
            rewardTMinus1s,
            visitCounts,
            gameMode,
            GRIDDIMENSION,
            targetPositions,
            LAMBDA,
            singleLearner,
            batches,
            IGNOREnumberAgents,
        ) = args

        print_policy(policies[0], NUMSTATES)
        totalvisitCount = jnp.sum(visitCounts, axis=0)

        visitTotal = jnp.sum(totalvisitCount)
        percentages = np.array(
            jnp.round((totalvisitCount / visitTotal) * 100), dtype=np.int8
        )
        print("visit percentages:")
        print_in_grid_shape(percentages, NUMSTATES)

        if algoType == "centralised":
            agent = 0
            if onlines[agent]:
                newQs, key = batch_learn(
                    batches[agent],
                    learningIterationsL,
                    learningRateBeta,
                    allQvalues[agent],
                    policies[agent],
                    GAMMA,
                    key,
                    LAMBDA,
                )
                allQvalues = allQvalues.at[agent].set(newQs)
            batches = jnp.empty((numberAgents, batchLength, 5))

        else:
            allQvalues, batches, key = batch_learn_for_all(
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
            )

        @jit
        def add_to_regDiscReturnLists(regDiscReturnList, regDiscReturn, k):
            return regDiscReturnList.at[k].set(regDiscReturn)

        averageReturn = jnp.sum(regDiscReturns) / numberAgents
        averageReturnList = averageReturnList.at[k].set(averageReturn)

        policyEta = eta

        if algoType == "centralised":
            if onlines[0]:
                newPol = PMA_update(
                    lowerBound,
                    policies[0],
                    allQvalues[0],
                    LAMBDA,
                    policyEta,
                    onlines[0],
                )
                policies = policies.at[0].set(newPol)

                newPolicies = vmap(copy_central, (0, None), 0)(
                    jnp.arange(numberAgents), policies
                )
                policies = newPolicies

            if testRobustness == "one_time_addition" and k == (oneTimeIncrease - 1):
                (
                    key,
                    numberAgents,
                    policies,
                    allQvalues,
                    visitCounts,
                    regDiscReturns,
                    sigmas,
                    stateTs,
                    actionTs,
                    rewardTs,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTPlus1s,
                    batches,
                    maxMpg,
                    onlines,
                ) = one_time_addition(
                    key,
                    numberSpareAgents,
                    numberAgents,
                    policies,
                    allQvalues,
                    visitCounts,
                    regDiscReturns,
                    sigmas,
                    stateTs,
                    actionTs,
                    rewardTs,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTPlus1s,
                    batches,
                    maxMpg,
                    onlines,
                )

        else:
            policies = update_all_policies(
                lowerBound, policies, allQvalues, LAMBDA, policyEta, onlines
            )

        if (
            testRobustness == "one_time_addition"
            and k == (oneTimeIncrease - 1)
            and algoType != "centralised"
        ):
            (
                key,
                numberAgents,
                policies,
                allQvalues,
                visitCounts,
                regDiscReturns,
                sigmas,
                stateTs,
                actionTs,
                rewardTs,
                stateTMinus1s,
                actionTMinus1s,
                rewardTMinus1s,
                stateTMinus2s,
                actionTMinus2s,
                rewardTMinus2s,
                stateTPlus1s,
                batches,
                maxMpg,
                onlines,
            ) = one_time_addition(
                key,
                numberSpareAgents,
                numberAgents,
                policies,
                allQvalues,
                visitCounts,
                regDiscReturns,
                sigmas,
                stateTs,
                actionTs,
                rewardTs,
                stateTMinus1s,
                actionTMinus1s,
                rewardTMinus1s,
                stateTMinus2s,
                actionTMinus2s,
                rewardTMinus2s,
                stateTPlus1s,
                batches,
                maxMpg,
                onlines,
            )

        if "networked" in algoType:
            if algoType == "networkedEvalPol":
                regDiscReturns = jnp.full(numberAgents, 0, jnp.float64)
                timeWithinK = 0
                (
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
                    regDiscReturns,
                    time,
                    timeWithinK,
                    key,
                ) = mtd_stepper(
                    evalIterations,
                    stateTs,
                    policies,
                    key,
                    GAMMA,
                    timeWithinK,
                    ACTIONS,
                    actionTs,
                    rewardTs,
                    regDiscReturns,
                    stateTPlus1s,
                    time,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    visitCounts,
                    gameMode,
                    GRIDDIMENSION,
                    targetPositions,
                    LAMBDA,
                )

                sigmas = jnp.multiply(regDiscReturns, onlines)

            uniquePolicies = np.unique(np.array(policies), axis=0)
            debug.print(str(len(uniquePolicies)))

            if temperature is None:  # stepped temperature annealing scheme is set here
                temperature = (10000 / 10 ** math.ceil((maxK - 1) / 10)) * (
                    10 ** math.ceil(k / 10)
                )

            @jit
            def do_sharing_iterations(iteration, args):
                (
                    temperature,
                    stateTs,
                    policies,
                    sigmas,
                    communicationRadius,
                    soft,
                    key,
                    GRIDDIMENSION,
                    GAMMA,
                    timeWithinK,
                    ACTIONS,
                    actionTs,
                    rewardTs,
                    regDiscReturns,
                    stateTPlus1s,
                    time,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    visitCounts,
                    gameMode,
                    targetPositions,
                    LAMBDA,
                ) = args
                policies, sigmas, key = shareAndAdoptPolicies(
                    stateTs,
                    policies,
                    sigmas,
                    communicationRadius,
                    soft,
                    key,
                    GRIDDIMENSION,
                    temperature,
                )

                (
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
                    regDiscReturns,
                    time,
                    timeWithinK,
                    key,
                ) = mtd_stepper(
                    1,
                    stateTs,
                    policies,
                    key,
                    GAMMA,
                    timeWithinK,
                    ACTIONS,
                    actionTs,
                    rewardTs,
                    regDiscReturns,
                    stateTPlus1s,
                    time,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    visitCounts,
                    gameMode,
                    GRIDDIMENSION,
                    targetPositions,
                    LAMBDA,
                )
                return (
                    temperature,
                    stateTs,
                    policies,
                    sigmas,
                    communicationRadius,
                    soft,
                    key,
                    GRIDDIMENSION,
                    GAMMA,
                    timeWithinK,
                    ACTIONS,
                    actionTs,
                    rewardTs,
                    regDiscReturns,
                    stateTPlus1s,
                    time,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    rewardTMinus1s,
                    visitCounts,
                    gameMode,
                    targetPositions,
                    LAMBDA,
                )

            args = (
                temperature,
                stateTs,
                policies,
                sigmas,
                communicationRadius,
                soft,
                key,
                GRIDDIMENSION,
                GAMMA,
                timeWithinK,
                ACTIONS,
                actionTs,
                rewardTs,
                regDiscReturns,
                stateTPlus1s,
                time,
                stateTMinus2s,
                actionTMinus2s,
                rewardTMinus2s,
                stateTMinus1s,
                actionTMinus1s,
                rewardTMinus1s,
                visitCounts,
                gameMode,
                targetPositions,
                LAMBDA,
            )
            args = fori_loop(0, maxSharingIterationsC, do_sharing_iterations, args)
            (
                temperature,
                stateTs,
                policies,
                sigmas,
                communicationRadius,
                soft,
                key,
                GRIDDIMENSION,
                GAMMA,
                timeWithinK,
                ACTIONS,
                actionTs,
                rewardTs,
                regDiscReturns,
                stateTPlus1s,
                time,
                stateTMinus2s,
                actionTMinus2s,
                rewardTMinus2s,
                stateTMinus1s,
                actionTMinus1s,
                rewardTMinus1s,
                visitCounts,
                gameMode,
                targetPositions,
                LAMBDA,
            ) = args
            uniquePolicies = np.unique(np.array(policies), axis=0)
            debug.print(str(len(uniquePolicies)))

    return averageReturnList, exploitabilityList, policyNorms


@jit
def get_empirical_distribution(policies, stateTs):
    statePlacements = vmap(get_statePlacement, (None, 0), 0)(policies, stateTs)
    state_distribution = jnp.sum(statePlacements, axis=0)
    state_distribution = jnp.divide(state_distribution, len(stateTs))
    return state_distribution


@jit
def get_statePlacement(policies, stateT):
    statePlacement = jnp.full(jnp.shape(policies)[1], 0)
    statePlacement = statePlacement.at[stateT].add(1)
    return statePlacement


# population's policy divergence as defined in Sec. 7.2.3.
@jit
def get_policy_norm(policies):
    maxDifferences = vmap(addToRunningTotal, (None, 0), 0)(policies, policies[1:])
    total = jnp.sum(maxDifferences)

    return total / jnp.shape(policies)[0]


@jit
def addToRunningTotal(policies, otherPolicy):
    differences = vmap(getMaxDifference, (0, None, None), 0)(
        jnp.arange(jnp.shape(policies)[1]), policies, otherPolicy
    )
    maxDifference = jnp.max(differences)

    return maxDifference


@jit
def getMaxDifference(state, policies, otherPolicy):
    difference = jnp.linalg.norm(
        jnp.subtract(policies[0][state], otherPolicy[state]), ord=1
    )
    return difference


@jit
def step_environment(iteration, args):
    (
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
    ) = args

    muHatT = get_empirical_distribution(policies, stateTs)

    actionTs, rewardTs, regDiscReturns, stateTPlus1s, key = step_all_agents(
        stateTs,
        policies,
        key,
        GAMMA,
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
    )
    time += 1
    timeWithinK += 1

    (
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
    ) = store_transitions(
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

    return (
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
    )


@jit
def copy_central(agent, policies):
    return policies[0]


@partial(jit, static_argnames=["maxMtd"])
def mtd_stepper(
    maxMtd,
    stateTs,
    policies,
    key,
    GAMMA,
    timeWithinK,
    ACTIONS,
    actionTs,
    rewardTs,
    regDiscReturns,
    stateTPlus1s,
    time,
    stateTMinus2s,
    actionTMinus2s,
    rewardTMinus2s,
    stateTMinus1s,
    actionTMinus1s,
    rewardTMinus1s,
    visitCounts,
    gameMode,
    GRIDDIMENSION,
    targetPositions,
    LAMBDA,
):
    args = (
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
    )
    args = fori_loop(0, maxMtd, step_environment, args)
    (
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
    ) = args
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
        regDiscReturns,
        time,
        timeWithinK,
        key,
    )


@jit
def mpg_step(mpg, args):
    (
        maxMtd,
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
        singleLearner,
        batches,
        numberAgents,
    ) = args

    (
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
        regDiscReturns,
        time,
        timeWithinK,
        key,
    ) = mtd_stepper(
        maxMtd,
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
    )

    @jit
    def check_single_learner():
        return jnp.select(
            [singleLearner == True, singleLearner == False],  # noqa: E712
            [
                batches.at[0, mpg - 1].set(
                    jnp.array(
                        [
                            stateTMinus2s[0],
                            actionTMinus2s[0],
                            rewardTMinus2s[0],
                            stateTMinus1s[0],
                            actionTMinus1s[0],
                        ]
                    )
                ),
                add_to_all_batches(
                    numberAgents,
                    stateTMinus2s,
                    actionTMinus2s,
                    rewardTMinus2s,
                    stateTMinus1s,
                    actionTMinus1s,
                    batches,
                    mpg - 1,
                ),
            ],
        )

    batches = jnp.select(
        [mpg > 0, mpg <= 0],
        [check_single_learner(), batches],
    )

    return (
        maxMtd,
        stateTs,
        policies,
        key,
        GAMMA,
        timeWithinK,
        ACTIONS,
        actionTs,
        rewardTs,
        regDiscReturns,
        stateTPlus1s,
        time,
        stateTMinus2s,
        actionTMinus2s,
        rewardTMinus2s,
        stateTMinus1s,
        actionTMinus1s,
        rewardTMinus1s,
        visitCounts,
        gameMode,
        GRIDDIMENSION,
        targetPositions,
        LAMBDA,
        singleLearner,
        batches,
        numberAgents,
    )
