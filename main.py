# ruff: noqa: F405
# ruff: noqa: F403
from algorithmsDriver import *
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import numpy as np


# conducts provided number of trials of the algorithm and saves the data from each run
def test_algorithm(
    experiment,
    trials,
    LAMBDA,
    GRIDDIMENSION,
    game,
    numberAgents,
    algoType,
    maxK,
    maxMpg,
    maxMtd,
    GAMMA,
    lh,
    eta,
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
    NUMACTIONS = 5

    label = ""

    listAverageReturnList = jnp.empty((trials, maxK))
    listExploitabilityList = []
    listPolicyNorms = jnp.empty((trials, maxK))
    run_times = []
    run_cpu_times = []

    for trial in range(trials):
        start_time = time.time()
        start_cpu_time = time.process_time()
        averageReturnList, exploitabilityList, policyNorms = run_algorithm(
            trial,
            game,
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
        )
        end_time = time.time()
        end_cpu_time = time.process_time()
        run_time = end_time - start_time
        run_cpu_time = end_cpu_time - start_cpu_time
        run_times.append(run_time)
        run_cpu_times.append(run_cpu_time)

        listAverageReturnList = listAverageReturnList.at[trial].set(averageReturnList)
        listExploitabilityList.append(exploitabilityList)
        listPolicyNorms = listPolicyNorms.at[trial].set(policyNorms)

        if game == 1:
            gameMode = "agree on a single target"

        elif game == 0:
            gameMode = "cluster"

        elif game == 2:
            gameMode = "diffuse and stay"

        elif game == 3:
            gameMode = "cover targets and stay"

        elif game == 4:
            gameMode = "beach bar and stay"

        elif game == 5:
            gameMode = "diffuse round ring and stay"

        elif game == 6:
            gameMode = "push box"

        elif game == 7:
            gameMode = "evade in cluster"

        thisRunLocation = "plots/" + gameMode + "/"
        if testRobustness is not None:
            thisRunLocation += testRobustness + "/"
        thisRunLocation += (
            "dimension = " + str(GRIDDIMENSION) + "/agents = " + str(numberAgents)
        )
        thisRunLocation += (
            "/"
            + "maxMpg = "
            + str(maxMpg)
            + "/"
            + "learningIterationsL = "
            + str(learningIterationsL)
        )
        thisRunLocation += (
            "/"
            + "maxSharingIterationsC = "
            + str(maxSharingIterationsC)
            + "/"
            + "evalIterations = "
            + str(evalIterations)
        )
        thisRunLocation += "/" + experiment + "/k = " + str(maxK)
        if testExploitation:
            thisRunLocation += "/Exploitation"
        else:
            thisRunLocation += "/No Exploitation"
        thisRunLocation += "/" + algoType
        if communicationFrac is not None:
            thisRunLocation += "; " + str(communicationFrac)
        if soft and ("networked" in algoType):
            thisRunLocation += "; SoftMax"
            if temperature is not None:
                thisRunLocation += " temperature = " + str(temperature)
            else:
                thisRunLocation += " temperature evolves"

        thisAlgoLocation = thisRunLocation

        thisRunLocation += "/run = " + str(trial + 1)
        cwd = os.getcwd()
        path = os.path.join(cwd, thisRunLocation)
        os.makedirs(path, exist_ok=True)

        with open(thisRunLocation + "/" + label + ".txt", "w") as f:
            print("", file=f)
            print("Number of agents = ", numberAgents, file=f)

            print("", file=f)
            print("#Run = ", trial, file=f)

            print(
                "listAverageReturnList.append(",
                np.array(averageReturnList).tolist(),
                ")",
                file=f,
            )

            print(
                "listExploitabilityList.append(",
                np.array(exploitabilityList).tolist(),
                ")",
                file=f,
            )

            print(
                "listPolicyNorms.append(", np.array(policyNorms).tolist(), ")", file=f
            )

            print("run_times.append(", np.array(run_time).tolist(), ")", file=f, sep="")

            print(
                "run_cpu_times.append(",
                np.array(run_cpu_time).tolist(),
                ")",
                file=f,
                sep="",
            )

        plt.clf()
        plt.subplot(3, 1, 2)

        plt.plot(averageReturnList)
        plt.ylabel("Average discounted return")

        if maxK > 150:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
        else:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

        exploitTestFrequency = 2
        plt.subplot(3, 1, 1)
        xaxis = [exploitTestFrequency * i for i in range(len(exploitabilityList))]
        plt.plot(xaxis, exploitabilityList)
        plt.ylabel("Exploitability")

        plt.subplot(3, 1, 3)
        plt.plot(policyNorms)
        plt.ylabel("Total policy diff.")

        plt.savefig(thisRunLocation + "/all: " + label + ".png")

        plt.clf()

        plt.plot(averageReturnList)
        plt.ylabel("Discounted regularised reward")
        plt.xlabel("Outer iterations (k)")

        if maxK > 150:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
        else:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

        plt.savefig(thisRunLocation + "/rewards: " + label + ".png")

        plt.clf()

        xaxis = [exploitTestFrequency * i for i in range(len(exploitabilityList))]
        plt.plot(xaxis, exploitabilityList)
        plt.ylabel("Exploitability")
        plt.xlabel("Outer iterations (k)")

        if maxK > 150:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
        else:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

        plt.savefig(thisRunLocation + "/exploitability: " + label + ".png")

        plt.clf()

        if maxK > 150:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
        else:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

        plt.savefig(thisRunLocation + "/policy differences: " + label + ".png")

        plt.clf()

        plt.plot(policyNorms)
        plt.ylabel("Average policy norm")
        plt.xlabel("Outer iterations (k)")

        if maxK > 150:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
        else:
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

        plt.savefig(thisRunLocation + "/policy norms: " + label + ".png")

    def get_mean_and_stDev(listOftrials):
        listOftrials = np.array(listOftrials)

        means = []
        stDevs = []
        for iteration in range(len(listOftrials[0])):
            runValuesAtIteration = []
            for run in range(len(listOftrials)):
                runValuesAtIteration.append(listOftrials[run][iteration])
            means.append(np.mean(runValuesAtIteration))
            stDevs.append(np.std(runValuesAtIteration))
        twostDevs = 2 * np.array(stDevs)

        return means, twostDevs.tolist()

    meanAverageReward, stdevAverageReward = get_mean_and_stDev(listAverageReturnList)
    meanExploitability, stdevExploitability = get_mean_and_stDev(listExploitabilityList)
    meanPolicyNorms, stdevPolicyNorms = get_mean_and_stDev(listPolicyNorms)
    mean_run_time = np.mean(np.array(run_times))
    stdev_run_time = 2 * np.std(np.array(run_times))
    mean_run_cpu_time = np.mean(np.array(run_cpu_times))
    stdev_run_cpu_time = 2 * np.std(np.array(run_cpu_times))

    thisAlgoLocation += "/Summary"
    cwd = os.getcwd()
    path = os.path.join(cwd, thisAlgoLocation)
    os.makedirs(path, exist_ok=True)

    with open(thisAlgoLocation + "/saving averages," + label + ".txt", "w") as f:
        if algoType == "centralised" or algoType == "independent":
            prefix = algoType
        else:
            prefix = "soft" + str(int(communicationFrac * 100)) + "pc"

        print(
            prefix,
            "meanAverageReward = np.array(",
            meanAverageReward,
            ")",
            file=f,
            sep="",
        )
        print(
            prefix,
            "stdevAverageReward = np.array(",
            stdevAverageReward,
            ")",
            file=f,
            sep="",
        )
        print(
            prefix,
            "meanExploitability = np.array(",
            meanExploitability,
            ")",
            file=f,
            sep="",
        )
        print(
            prefix,
            "stdevExploitability = np.array(",
            stdevExploitability,
            ")",
            file=f,
            sep="",
        )
        print(
            prefix, "meanPolicyNorms = np.array(", meanPolicyNorms, ")", file=f, sep=""
        )
        print(
            prefix,
            "stdevPolicyNorms = np.array(",
            stdevPolicyNorms,
            ")",
            file=f,
            sep="",
        )
        print("run_times = np.array(", run_times, ")", file=f, sep="")
        print("mean_run_time = np.array(", mean_run_time, ")", file=f, sep="")
        print("stdev_run_time = np.array(", stdev_run_time, ")", file=f, sep="")
        print("run_cpu_times = np.array(", run_cpu_times, ")", file=f, sep="")
        print("mean_run_cpu_time = np.array(", mean_run_cpu_time, ")", file=f, sep="")
        print("stdev_run_cpu_time = np.array(", stdev_run_cpu_time, ")", file=f, sep="")

    plt.clf()
    plt.subplot(3, 1, 2)
    plt.plot(meanAverageReward)
    plt.ylabel("Average discounted return")

    if maxK > 150:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
    else:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.subplot(3, 1, 1)
    xaxis = [exploitTestFrequency * i for i in range(len(exploitabilityList))]
    plt.plot(xaxis, meanExploitability)
    plt.ylabel("Exploitability")

    plt.subplot(3, 1, 3)
    plt.plot(meanPolicyNorms)
    plt.ylabel("Total policy diff.")

    plt.savefig(thisAlgoLocation + "/all: " + label + ".png")

    plt.clf()

    plt.plot(meanAverageReward)
    plt.ylabel("Discounted regularised reward")
    plt.xlabel("Outer iterations (k)")

    if maxK > 150:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
    else:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.savefig(thisAlgoLocation + "/rewards: " + label + ".png")

    plt.clf()

    xaxis = [exploitTestFrequency * i for i in range(len(exploitabilityList))]
    plt.plot(xaxis, meanExploitability)
    plt.ylabel("Exploitability")
    plt.xlabel("Outer iterations (k)")

    if maxK > 150:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
    else:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.savefig(thisAlgoLocation + "/exploitability: " + label + ".png")

    plt.clf()

    plt.plot(meanPolicyNorms)
    plt.ylabel("Average policy norm")
    plt.xlabel("Outer iterations (k)")

    if maxK > 150:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
    else:
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.savefig(thisAlgoLocation + "/policy norms: " + label + ".png")

    return (
        jnp.array(meanAverageReward),
        jnp.array(stdevAverageReward),
        jnp.array(meanExploitability),
        jnp.array(stdevExploitability),
        jnp.array(meanPolicyNorms),
        jnp.array(stdevPolicyNorms),
    )


if __name__ == "__main__":
    gameModes = ["cluster", "agree on a single target"]
    testRobustnesses = [None, "continued_random_failures", "one_time_addition"]
    gridDimensions = [8, 16]
    temperatures = [
        None,
        100,
    ]  # 'temperature = None' gives temperature annealing scheme as defined in file 'algorithmsDriver.py'

    for temperature in temperatures:
        for gridDimension in gridDimensions:
            for testRobustness in testRobustnesses:
                for gameMode in gameModes:
                    if gameMode == "agree on a single target":
                        game = 1
                    elif gameMode == "cluster":
                        game = 0

                    oneTimeIncrease = 300
                    if testRobustness != "one_time_addition":
                        oneTimeIncrease = None

                    maxK = 200
                    if testRobustness == "one_time_addition" or gridDimension == 16:
                        maxK = 400
                    numStates = gridDimension**2
                    trials = 10
                    numberAgents = 250  # when testing for robustness to population increase, the population will start at 1/5 of this value before the remaining 4/5 of the population is added
                    numActions = 5
                    if gridDimension == 8:
                        maxMpg = 500
                    elif gridDimension == 16:
                        maxMpg = 1000
                    maxMtd = 1
                    maxSharingIterationsC = 1
                    learningIterationsL = 100
                    evalIterations = 100
                    gamma = 0.9
                    learningRateBeta = 0.1
                    eta = 0.01
                    LAMBDA = 0.00

                    # Lipschitz constants as in Assumption 3.9 required theoretically for Definition 4.2; in practice we ignore this bound as discussed in Sec. 7.4.1
                    ks = 2
                    ka = 2
                    assert (ks >= 0) and (ks <= 2), "ks out of bounds"
                    assert (ka >= 0) and (ka <= 2), "ka out of bounds"
                    ls = 1
                    la = 1
                    assert (ls >= 0) and (ls <= 1), "ls out of bounds"
                    assert (la >= 0) and (la <= 1), "la out of bounds"
                    lh = la + (
                        (gamma * ls * ka) / (2 - (gamma * ks))
                    )  # L_h as in our Definition 4.2

                    soft = True  # setting 'soft' to False gives a max function for policy adoption rather than our softmax scheme
                    testExploitation = True  # setting this to False turns off the exploitation approximation steps

                    label = ""
                    experiment = ""

                    start_all_time = time.time()
                    start_all_cpu_time = time.process_time()

                    algoType = "independent"
                    communicationFrac = None
                    (
                        independentmeanAverageReward,
                        independentstdevAverageReward,
                        independentmeanExploitability,
                        independentstdevExploitability,
                        independentmeanPolicyNorms,
                        independentstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    algoType = "centralised"
                    communicationFrac = None
                    (
                        centralisedmeanAverageReward,
                        centralisedstdevAverageReward,
                        centralisedmeanExploitability,
                        centralisedstdevExploitability,
                        centralisedmeanPolicyNorms,
                        centralisedstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    algoType = "networkedEvalPol"
                    communicationFrac = 0.2
                    (
                        soft20pcmeanAverageReward,
                        soft20pcstdevAverageReward,
                        soft20pcmeanExploitability,
                        soft20pcstdevExploitability,
                        soft20pcmeanPolicyNorms,
                        soft20pcstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    algoType = "networkedEvalPol"
                    communicationFrac = 0.4
                    (
                        soft40pcmeanAverageReward,
                        soft40pcstdevAverageReward,
                        soft40pcmeanExploitability,
                        soft40pcstdevExploitability,
                        soft40pcmeanPolicyNorms,
                        soft40pcstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    algoType = "networkedEvalPol"
                    communicationFrac = 0.6
                    (
                        soft60pcmeanAverageReward,
                        soft60pcstdevAverageReward,
                        soft60pcmeanExploitability,
                        soft60pcstdevExploitability,
                        soft60pcmeanPolicyNorms,
                        soft60pcstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    algoType = "networkedEvalPol"
                    communicationFrac = 0.8
                    (
                        soft80pcmeanAverageReward,
                        soft80pcstdevAverageReward,
                        soft80pcmeanExploitability,
                        soft80pcstdevExploitability,
                        soft80pcmeanPolicyNorms,
                        soft80pcstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    algoType = "networkedEvalPol"
                    communicationFrac = 1.0
                    (
                        soft100pcmeanAverageReward,
                        soft100pcstdevAverageReward,
                        soft100pcmeanExploitability,
                        soft100pcstdevExploitability,
                        soft100pcmeanPolicyNorms,
                        soft100pcstdevPolicyNorms,
                    ) = test_algorithm(
                        experiment,
                        trials,
                        LAMBDA,
                        gridDimension,
                        game,
                        numberAgents,
                        algoType,
                        maxK,
                        maxMpg,
                        maxMtd,
                        gamma,
                        lh,
                        eta,
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
                    )

                    end_all_time = time.time()
                    end_all_cpu_time = time.process_time()

                    all_time = end_all_time - start_all_time
                    all_cpu_time = end_all_cpu_time - start_all_cpu_time

                    thisExpLocation = "plots/" + gameMode + "/"
                    if testRobustness is not None:
                        thisExpLocation += testRobustness + "/"
                    thisExpLocation += (
                        "dimension = "
                        + str(gridDimension)
                        + "/agents = "
                        + str(numberAgents)
                    )
                    thisExpLocation += (
                        "/"
                        + "maxMpg = "
                        + str(maxMpg)
                        + "/"
                        + "learningIterationsL = "
                        + str(learningIterationsL)
                    )
                    thisExpLocation += (
                        "/"
                        + "maxSharingIterationsC = "
                        + str(maxSharingIterationsC)
                        + "/"
                        + "evalIterations = "
                        + str(evalIterations)
                    )
                    thisExpLocation += "/" + experiment + "/k = " + str(maxK)
                    if testExploitation:
                        thisExpLocation += "/Exploitation"
                    else:
                        thisExpLocation += "/No Exploitation"
                    thisExpLocation += "/Summary"

                    cwd = os.getcwd()
                    path = os.path.join(cwd, thisExpLocation)
                    os.makedirs(path, exist_ok=True)

                    with open(thisExpLocation + "/time," + label + ".txt", "w") as f:
                        print("all_time = ", all_time, file=f)
                        print("all_cpu_time = ", all_cpu_time, file=f)

                    line_styles = [
                        "solid",
                        (0, (1, 1)),
                        "dashed",
                        "dashdot",
                        ":",
                        (0, (3, 1, 1, 1)),
                        (0, (5, 1)),
                    ]
                    markers = [",", ",", ",", ",", ",", ",", ","]
                    hatch_patterns = ["//", "\\", "||", "--", "..", "OO", "oo"]

                    plt.clf()
                    plt.figure(figsize=(10, 4.5))  # default 6.4 wide and 4.8 high
                    if (testRobustness is not None) and ("one_time_" in testRobustness):
                        plt.axvline(
                            oneTimeIncrease, 0, 1, color="black", linestyle="dashed"
                        )
                    plt.plot(
                        centralisedmeanAverageReward,
                        label="Centralised",
                        linestyle=line_styles[0],
                        marker=markers[0],
                    )
                    plt.fill_between(
                        range(len(centralisedmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                centralisedmeanAverageReward,
                                centralisedstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                centralisedmeanAverageReward,
                                centralisedstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[0]
                    )
                    plt.plot(
                        independentmeanAverageReward,
                        label="Independent",
                        linestyle=line_styles[1],
                        marker=markers[1],
                    )
                    plt.fill_between(
                        range(len(independentmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                independentmeanAverageReward,
                                independentstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                independentmeanAverageReward,
                                independentstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[1]
                    )

                    plt.plot(
                        soft20pcmeanAverageReward,
                        label="Networked (0.2)",
                        linestyle=line_styles[2],
                        marker=markers[2],
                    )
                    plt.fill_between(
                        range(len(soft20pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft20pcmeanAverageReward,
                                soft20pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft20pcmeanAverageReward,
                                soft20pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[2]
                    )

                    plt.plot(
                        soft40pcmeanAverageReward,
                        label="Networked (0.4)",
                        linestyle=line_styles[3],
                        marker=markers[3],
                    )
                    plt.fill_between(
                        range(len(soft40pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft40pcmeanAverageReward,
                                soft40pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft40pcmeanAverageReward,
                                soft40pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[3]
                    )

                    plt.plot(
                        soft60pcmeanAverageReward,
                        label="Networked (0.6)",
                        linestyle=line_styles[4],
                        marker=markers[4],
                    )
                    plt.fill_between(
                        range(len(soft60pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft60pcmeanAverageReward,
                                soft60pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft60pcmeanAverageReward,
                                soft60pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[4]
                    )

                    plt.plot(
                        soft80pcmeanAverageReward,
                        label="Networked (0.8)",
                        linestyle=line_styles[5],
                        marker=markers[5],
                    )
                    plt.fill_between(
                        range(len(soft80pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft80pcmeanAverageReward,
                                soft80pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft80pcmeanAverageReward,
                                soft80pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[5]
                    )

                    plt.plot(
                        soft100pcmeanAverageReward,
                        label="Networked (1.0)",
                        linestyle=line_styles[6],
                        marker=markers[6],
                    )
                    plt.fill_between(
                        range(len(soft100pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft100pcmeanAverageReward,
                                soft100pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft100pcmeanAverageReward,
                                soft100pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[6]
                    )

                    plt.legend()
                    plt.ylabel("Discounted regularised reward")
                    plt.xlabel("k")
                    if maxK > 150:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
                    else:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
                    plt.tight_layout()
                    plt.savefig(
                        thisExpLocation + "/Combined, Rewards: " + label + ".png"
                    )

                    plt.clf()
                    plt.figure(figsize=(10, 4.5))  # default 6.4 wide and 4.8 high
                    if (testRobustness is not None) and ("one_time_" in testRobustness):
                        plt.axvline(
                            oneTimeIncrease, 0, 1, color="black", linestyle="dashed"
                        )
                    exploitTestFrequency = 2
                    xaxis = [
                        exploitTestFrequency * i
                        for i in range(len(soft20pcmeanExploitability))
                    ]
                    plt.plot(
                        xaxis,
                        centralisedmeanExploitability,
                        label="Centralised",
                        linestyle=line_styles[0],
                        marker=markers[0],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(
                            centralisedmeanExploitability,
                            centralisedstdevExploitability,
                        ),
                        jnp.subtract(
                            centralisedmeanExploitability,
                            centralisedstdevExploitability,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[0]
                    )
                    plt.plot(
                        xaxis,
                        independentmeanExploitability,
                        label="Independent",
                        linestyle=line_styles[1],
                        marker=markers[1],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(
                            independentmeanExploitability,
                            independentstdevExploitability,
                        ),
                        jnp.subtract(
                            independentmeanExploitability,
                            independentstdevExploitability,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[1]
                    )

                    plt.plot(
                        xaxis,
                        soft20pcmeanExploitability,
                        label="Networked (0.2)",
                        linestyle=line_styles[2],
                        marker=markers[2],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft20pcmeanExploitability, soft20pcstdevExploitability),
                        jnp.subtract(
                            soft20pcmeanExploitability, soft20pcstdevExploitability
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[2]
                    )

                    plt.plot(
                        xaxis,
                        soft40pcmeanExploitability,
                        label="Networked (0.4)",
                        linestyle=line_styles[3],
                        marker=markers[3],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft40pcmeanExploitability, soft40pcstdevExploitability),
                        jnp.subtract(
                            soft40pcmeanExploitability, soft40pcstdevExploitability
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[3]
                    )

                    plt.plot(
                        xaxis,
                        soft60pcmeanExploitability,
                        label="Networked (0.6)",
                        linestyle=line_styles[4],
                        marker=markers[4],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft60pcmeanExploitability, soft60pcstdevExploitability),
                        jnp.subtract(
                            soft60pcmeanExploitability, soft60pcstdevExploitability
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[4]
                    )

                    plt.plot(
                        xaxis,
                        soft80pcmeanExploitability,
                        label="Networked (0.8)",
                        linestyle=line_styles[5],
                        marker=markers[5],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft80pcmeanExploitability, soft80pcstdevExploitability),
                        jnp.subtract(
                            soft80pcmeanExploitability, soft80pcstdevExploitability
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[5]
                    )

                    plt.plot(
                        xaxis,
                        soft100pcmeanExploitability,
                        label="Networked (1.0)",
                        linestyle=line_styles[6],
                        marker=markers[6],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(
                            soft100pcmeanExploitability,
                            soft100pcstdevExploitability,
                        ),
                        jnp.subtract(
                            soft100pcmeanExploitability,
                            soft100pcstdevExploitability,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[6]
                    )

                    plt.legend()
                    plt.ylabel("Exploitability")
                    plt.xlabel("k")
                    if maxK > 150:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
                    else:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
                    plt.tight_layout()
                    plt.savefig(
                        thisExpLocation + "/Combined, Exploitability: " + label + ".png"
                    )

                    plt.clf()
                    plt.figure(figsize=(10, 4.5))  # default 6.4 wide and 4.8 high
                    if (testRobustness is not None) and ("one_time_" in testRobustness):
                        plt.axvline(
                            oneTimeIncrease, 0, 1, color="black", linestyle="dashed"
                        )
                    plt.plot(
                        centralisedmeanPolicyNorms,
                        label="Centralised",
                        marker=markers[0],
                    )
                    plt.fill_between(
                        range(len(centralisedmeanPolicyNorms)),
                        jnp.add(
                            centralisedmeanPolicyNorms, centralisedstdevPolicyNorms
                        ),
                        jnp.subtract(
                            centralisedmeanPolicyNorms, centralisedstdevPolicyNorms
                        ),
                        alpha=0.3,
                    )
                    plt.plot(
                        independentmeanPolicyNorms,
                        label="Independent",
                        marker=markers[1],
                    )
                    plt.fill_between(
                        range(len(independentmeanPolicyNorms)),
                        np.add(independentmeanPolicyNorms, independentstdevPolicyNorms),
                        jnp.subtract(
                            independentmeanPolicyNorms, independentstdevPolicyNorms
                        ),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft20pcmeanPolicyNorms,
                        label="Networked (0.2)",
                        marker=markers[2],
                    )
                    plt.fill_between(
                        range(len(soft20pcmeanPolicyNorms)),
                        np.add(soft20pcmeanPolicyNorms, soft20pcstdevPolicyNorms),
                        jnp.subtract(soft20pcmeanPolicyNorms, soft20pcstdevPolicyNorms),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft40pcmeanPolicyNorms,
                        label="Networked (0.4)",
                        marker=markers[3],
                    )
                    plt.fill_between(
                        range(len(soft40pcmeanPolicyNorms)),
                        np.add(soft40pcmeanPolicyNorms, soft40pcstdevPolicyNorms),
                        jnp.subtract(soft40pcmeanPolicyNorms, soft40pcstdevPolicyNorms),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft60pcmeanPolicyNorms,
                        label="Networked (0.6)",
                        marker=markers[4],
                    )
                    plt.fill_between(
                        range(len(soft60pcmeanPolicyNorms)),
                        np.add(soft60pcmeanPolicyNorms, soft60pcstdevPolicyNorms),
                        jnp.subtract(soft60pcmeanPolicyNorms, soft60pcstdevPolicyNorms),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft80pcmeanPolicyNorms,
                        label="Networked (0.8)",
                        marker=markers[5],
                    )
                    plt.fill_between(
                        range(len(soft80pcmeanPolicyNorms)),
                        np.add(soft80pcmeanPolicyNorms, soft80pcstdevPolicyNorms),
                        jnp.subtract(soft80pcmeanPolicyNorms, soft80pcstdevPolicyNorms),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft100pcmeanPolicyNorms,
                        label="Networked (1.0)",
                        marker=markers[6],
                    )
                    plt.fill_between(
                        range(len(soft100pcmeanPolicyNorms)),
                        np.add(soft100pcmeanPolicyNorms, soft100pcstdevPolicyNorms),
                        jnp.subtract(
                            soft100pcmeanPolicyNorms, soft100pcstdevPolicyNorms
                        ),
                        alpha=0.3,
                    )

                    plt.legend()
                    plt.ylabel("Average policy norm")
                    plt.xlabel("k")
                    if maxK > 150:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(20))
                    else:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

                    plt.savefig(
                        thisExpLocation + "/Combined, Policy norms: " + label + ".png"
                    )

                    plt.clf()
                    plt.figure(figsize=(11, 8))
                    plt.subplot(3, 1, 2)
                    plt.plot(
                        centralisedmeanAverageReward,
                        label="Centralised",
                        linestyle=line_styles[0],
                        marker=markers[0],
                    )
                    plt.fill_between(
                        range(len(centralisedmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                centralisedmeanAverageReward,
                                centralisedstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                centralisedmeanAverageReward,
                                centralisedstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[0]
                    )
                    plt.plot(
                        independentmeanAverageReward,
                        label="Independent",
                        linestyle=line_styles[1],
                        marker=markers[1],
                    )
                    plt.fill_between(
                        range(len(independentmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                independentmeanAverageReward,
                                independentstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                independentmeanAverageReward,
                                independentstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[1]
                    )

                    plt.plot(
                        soft20pcmeanAverageReward,
                        label="Networked (0.2)",
                        linestyle=line_styles[2],
                        marker=markers[2],
                    )
                    plt.fill_between(
                        range(len(soft20pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft20pcmeanAverageReward,
                                soft20pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft20pcmeanAverageReward,
                                soft20pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[2]
                    )

                    plt.plot(
                        soft40pcmeanAverageReward,
                        label="Networked (0.4)",
                        linestyle=line_styles[3],
                        marker=markers[3],
                    )
                    plt.fill_between(
                        range(len(soft40pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft40pcmeanAverageReward,
                                soft40pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft40pcmeanAverageReward,
                                soft40pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[3]
                    )

                    plt.plot(
                        soft60pcmeanAverageReward,
                        label="Networked (0.6)",
                        linestyle=line_styles[4],
                        marker=markers[4],
                    )
                    plt.fill_between(
                        range(len(soft60pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft60pcmeanAverageReward,
                                soft60pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft60pcmeanAverageReward,
                                soft60pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[4]
                    )

                    plt.plot(
                        soft80pcmeanAverageReward,
                        label="Networked (0.8)",
                        linestyle=line_styles[5],
                        marker=markers[5],
                    )
                    plt.fill_between(
                        range(len(soft80pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft80pcmeanAverageReward,
                                soft80pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft80pcmeanAverageReward,
                                soft80pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[5]
                    )

                    plt.plot(
                        soft100pcmeanAverageReward,
                        label="Networked (1.0)",
                        linestyle=line_styles[6],
                        marker=markers[6],
                    )
                    plt.fill_between(
                        range(len(soft100pcmeanAverageReward)),
                        jnp.minimum(
                            jnp.add(
                                soft100pcmeanAverageReward,
                                soft100pcstdevAverageReward,
                            ),
                            10,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft100pcmeanAverageReward,
                                soft100pcstdevAverageReward,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[6]
                    )

                    plt.ylim(0, 10)
                    plt.legend()
                    # plt.legend().set_visible(False)
                    if (testRobustness is not None) and ("one_time_" in testRobustness):
                        plt.axvline(
                            oneTimeIncrease, 0, 1, color="black", linestyle="dashed"
                        )
                    plt.ylabel("Average discounted return")
                    plt.xlabel("k")

                    plt.subplot(3, 1, 1)
                    xaxis = [
                        exploitTestFrequency * i
                        for i in range(len(soft20pcmeanExploitability))
                    ]
                    plt.plot(
                        xaxis,
                        centralisedmeanExploitability,
                        label="Centralised",
                        linestyle=line_styles[0],
                        marker=markers[0],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(
                            centralisedmeanExploitability,
                            centralisedstdevExploitability,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                centralisedmeanExploitability,
                                centralisedstdevExploitability,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[0],
                    )
                    plt.plot(
                        xaxis,
                        independentmeanExploitability,
                        label="Independent",
                        linestyle=line_styles[1],
                        marker=markers[1],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(
                            independentmeanExploitability,
                            independentstdevExploitability,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                independentmeanExploitability,
                                independentstdevExploitability,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[1],
                    )

                    plt.plot(
                        xaxis,
                        soft20pcmeanExploitability,
                        label="Networked (0.2)",
                        linestyle=line_styles[2],
                        marker=markers[2],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft20pcmeanExploitability, soft20pcstdevExploitability),
                        jnp.maximum(
                            jnp.subtract(
                                soft20pcmeanExploitability, soft20pcstdevExploitability
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[2],alpha=0.3,
                    )

                    plt.plot(
                        xaxis,
                        soft40pcmeanExploitability,
                        label="Networked (0.4)",
                        linestyle=line_styles[3],
                        marker=markers[3],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft40pcmeanExploitability, soft40pcstdevExploitability),
                        jnp.maximum(
                            jnp.subtract(
                                soft40pcmeanExploitability, soft40pcstdevExploitability
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[3],alpha=0.3,
                    )

                    plt.plot(
                        xaxis,
                        soft60pcmeanExploitability,
                        label="Networked (0.6)",
                        linestyle=line_styles[4],
                        marker=markers[4],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft60pcmeanExploitability, soft60pcstdevExploitability),
                        jnp.maximum(
                            jnp.subtract(
                                soft60pcmeanExploitability, soft60pcstdevExploitability
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[4],alpha=0.3,
                    )

                    plt.plot(
                        xaxis,
                        soft80pcmeanExploitability,
                        label="Networked (0.8)",
                        linestyle=line_styles[5],
                        marker=markers[5],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(soft80pcmeanExploitability, soft80pcstdevExploitability),
                        jnp.maximum(
                            jnp.subtract(
                                soft80pcmeanExploitability, soft80pcstdevExploitability
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[5],alpha=0.3,
                    )

                    plt.plot(
                        xaxis,
                        soft100pcmeanExploitability,
                        label="Networked (1.0)",
                        linestyle=line_styles[6],
                        marker=markers[6],
                    )
                    plt.fill_between(
                        xaxis,
                        np.add(
                            soft100pcmeanExploitability,
                            soft100pcstdevExploitability,
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                soft100pcmeanExploitability,
                                soft100pcstdevExploitability,
                            ),
                            0,
                        ),
                        alpha=0.3,  # hatch=hatch_patterns[6],alpha=0.3
                    )
                    plt.ylim(0, 6)
                    plt.legend()
                    plt.legend().set_visible(False)
                    if (testRobustness is not None) and ("one_time_" in testRobustness):
                        plt.axvline(
                            oneTimeIncrease, 0, 1, color="black", linestyle="dashed"
                        )
                    plt.ylabel("Exploitability")
                    plt.xlabel("k")

                    plt.subplot(3, 1, 3)
                    plt.plot(
                        centralisedmeanPolicyNorms,
                        label="Centralised",
                        linestyle=line_styles[0],
                    )
                    plt.fill_between(
                        range(len(centralisedmeanPolicyNorms)),
                        jnp.add(
                            centralisedmeanPolicyNorms, centralisedstdevPolicyNorms
                        ),
                        jnp.maximum(
                            jnp.subtract(
                                centralisedmeanPolicyNorms, centralisedstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )
                    plt.plot(
                        independentmeanPolicyNorms,
                        label="Independent",
                        linestyle=line_styles[1],
                    )
                    plt.fill_between(
                        range(len(independentmeanPolicyNorms)),
                        np.add(independentmeanPolicyNorms, independentstdevPolicyNorms),
                        jnp.maximum(
                            jnp.subtract(
                                independentmeanPolicyNorms, independentstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft20pcmeanPolicyNorms,
                        label="Networked (0.2)",
                        linestyle=line_styles[2],
                    )
                    plt.fill_between(
                        range(len(soft20pcmeanPolicyNorms)),
                        np.add(soft20pcmeanPolicyNorms, soft20pcstdevPolicyNorms),
                        jnp.maximum(
                            jnp.subtract(
                                soft20pcmeanPolicyNorms, soft20pcstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft40pcmeanPolicyNorms,
                        label="Networked (0.4)",
                        linestyle=line_styles[3],
                    )
                    plt.fill_between(
                        range(len(soft40pcmeanPolicyNorms)),
                        np.add(soft40pcmeanPolicyNorms, soft40pcstdevPolicyNorms),
                        jnp.maximum(
                            jnp.subtract(
                                soft40pcmeanPolicyNorms, soft40pcstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft60pcmeanPolicyNorms,
                        label="Networked (0.6)",
                        linestyle=line_styles[4],
                    )
                    plt.fill_between(
                        range(len(soft60pcmeanPolicyNorms)),
                        np.add(soft60pcmeanPolicyNorms, soft60pcstdevPolicyNorms),
                        jnp.maximum(
                            jnp.subtract(
                                soft60pcmeanPolicyNorms, soft60pcstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft80pcmeanPolicyNorms,
                        label="Networked (0.8)",
                        linestyle=line_styles[5],
                    )
                    plt.fill_between(
                        range(len(soft80pcmeanPolicyNorms)),
                        np.add(soft80pcmeanPolicyNorms, soft80pcstdevPolicyNorms),
                        jnp.maximum(
                            jnp.subtract(
                                soft80pcmeanPolicyNorms, soft80pcstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )

                    plt.plot(
                        soft100pcmeanPolicyNorms,
                        label="Networked (1.0)",
                        linestyle=line_styles[6],
                    )
                    plt.fill_between(
                        range(len(soft100pcmeanPolicyNorms)),
                        np.add(soft100pcmeanPolicyNorms, soft100pcstdevPolicyNorms),
                        jnp.maximum(
                            jnp.subtract(
                                soft100pcmeanPolicyNorms, soft100pcstdevPolicyNorms
                            ),
                            0,
                        ),
                        alpha=0.3,
                    )

                    plt.ylim(0, 0.8)
                    plt.legend()
                    plt.legend().set_visible(False)
                    if (testRobustness is not None) and ("one_time_" in testRobustness):
                        plt.axvline(
                            oneTimeIncrease, 0, 1, color="black", linestyle="dashed"
                        )
                    plt.ylabel("Policy divergence")
                    plt.xlabel("k")

                    fig = plt.gcf()
                    fig.align_ylabels()
                    plt.tight_layout()

                    plt.savefig(thisExpLocation + "/Combined, All: " + label + ".png")
                    plt.clf()
