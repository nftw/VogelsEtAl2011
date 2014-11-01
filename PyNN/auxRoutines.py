# ###########################################
#
# Model implementation converted from
# Brain to PyNN by Vitor Chaud,
# Andrew Davison and Padraig Gleeson(2013).
#
# Original implementation reference:
#
# Inhibitory synaptic plasticity in a
# recurrent network model (F. Zenke, 2011)
#
# Adapted from:
# Vogels, T. P., H. Sprekeler, F. Zenke,
#   C. Clopath, and W. Gerstner. 'Inhibitory
#   Plasticity Balances Excitation and
#   Inhibition in Sensory Pathways and
#   Memory Networks.' Science (November 10, 2011).
#
# ###########################################

########################################################################
##
##  Auxiliary Routines
##
########################################################################


import numpy as np
import matplotlib.pyplot as plt


def get_neuron_isis(neuron_spike_times):
    """Gets the interspike intervals of a neurons within a population"""
    if neuron_spike_times.size == 0 or neuron_spike_times.size == 1:
        return -1 * np.ones(1)
    else:
        return np.diff(neuron_spike_times)


def calculate_neuron_firing_rate(neuron_spike_times):
    "Calculate firing rate based on the inverse of ISIs. Result in Hz."
    neuron_isis = get_neuron_isis(neuron_spike_times)
    if neuron_isis[0] == -1:
        ifr = 0
    else:
        ifr = 1000 / neuron_isis
    return np.mean(ifr)


def calculate_isicv2(neuron_spikes):
    neuron_isis = get_neuron_isis(neuron_spikes)
    if neuron_isis[0] == -1:
        return -1
    else:
        return np.std(neuron_isis) / np.mean(neuron_isis)


def plotRaster2(popSpikes, color):
    seg = popSpikes.segments[0]
    for spiketrain in seg.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.', c=color)
    plt.xlabel('Time [ms]')


def plot_raster(axis, spike_trains, color, y_axis_factor):
    i = 0
    for spike_train in spike_trains:
        i += 1
        y = y_axis_factor * np.ones_like(spike_train) * i
        plt.plot(spike_train, y, '.', c=color)
    plt.xlabel('Time [ms]')
    axis.spines['top'].set_color('none')
    axis.spines['left'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.tick_params(axis='x', top='off')
    axis.tick_params(axis='x', bottom='off')
    axis.tick_params(axis='y', left='off')

    axis.tick_params(axis='y', right='off')
    axis.spines['bottom'].set_linewidth(2)
    axis.spines['left'].set_linewidth(2)


def plotSpikeTrains(popSpikes, timeStep, simTimeIni, simTimeFin):
    seg = popSpikes.segments[0]
    for neuronSpikeTimes in seg.spiketrains:
        y = createSpikeTrain(neuronSpikeTimes, timeStep, simTimeIni, simTimeFin)
        x = np.linspace(simTimeIni, simTimeFin, num=int((simTimeFin - simTimeIni) / timeStep))
        plt.plot(x, y)
    plt.xlabel('Time [ms]')


def plotFilteredSpikeTrains(popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
    seg = popSpikes.segments[0]
    for neuronSpikeTimes in seg.spiketrains:
        y = filterSpikesTrain(neuronSpikeTimes, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
        numPoints = int((simTimeFin - simTimeIni) / timeStep)
        if int(2 * timeBoundKernel / timeStep) + 1 > numPoints:
            numPoints = int(2 * timeBoundKernel / timeStep) + 1
        x = np.linspace(simTimeIni, simTimeFin, num=numPoints)
        plt.plot(x, y)
    plt.xlabel('Time [ms]')


def plot_histogram(bar_color, spike_trains):
    isicvs = np.zeros(0)
    for neuron_spikes in spike_trains:
        neuron_isicv = calculate_isicv2(neuron_spikes)
        if neuron_isicv != -1:
            isicvs = np.append(isicvs, neuron_isicv)
    if np.size(isicvs) != 0:
        plt.hist(isicvs, histtype='stepfilled', color=bar_color, alpha=0.6)


def plot_isicv_double_hist(axis, pop_spike_trains, bar_color, pop_spike_trains2, bar_color2):
    plot_histogram(bar_color2, pop_spike_trains2)
    plot_histogram(bar_color, pop_spike_trains)
    plt.xlabel('ISI CV')
    plt.xlim((0.0, 3.0))
    axis.spines['top'].set_color('none')
    axis.spines['left'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.tick_params(axis='x', top='off')
    axis.tick_params(axis='x', bottom='off')
    axis.tick_params(axis='y', left='off')
    axis.tick_params(axis='y', right='off')
    axis.spines['bottom'].set_linewidth(2)
    axis.spines['left'].set_linewidth(2)


def biExponentialKernelFunction(t):
    "Calculate the bi-exponential kernel as in Vogels et al. 2011. The time is specified in ms"
    tau1 = 50  # [ms]
    tau2 = 4 * tau1  # [ms]
    return (1. / tau1) * np.exp(-np.absolute(t) / tau1) - (1. / tau2) * np.exp(-np.absolute(t) / tau2)


def biExponentialKernel(timeStep, timeBoundKernel):
    "Calculate the bi-exponential kernel as in Vogels et al. 2011. The time is specified in ms"
    numOfPoints = int(2 * timeBoundKernel / timeStep) + 1
    kernel = np.zeros(numOfPoints)
    index = 0
    for t in np.linspace(-timeBoundKernel, timeBoundKernel, num=numOfPoints):
        kernel[index] = biExponentialKernelFunction(t)
        index += 1
    return kernel


def createSpikeTrain(neuronSpikes, timeStep, simTimeIni, simTimeFin):
    "Receives the spike times of a neuron and returns an array representing the spike train"
    spikeIndex = 0
    spikeTrain = np.zeros(int((simTimeFin - simTimeIni) / timeStep))
    for t in np.linspace(simTimeIni, simTimeFin, num=int((simTimeFin - simTimeIni) / timeStep)):
        if spikeIndex < np.size(neuronSpikes):
            spikeTime = neuronSpikes[spikeIndex]
            #print (t)
            #print (spikeTime)
            if int(t) == int(spikeTime):  #rounding to 1ms timeStep
                index = int((t - simTimeIni) / timeStep)
                spikeTrain[index] = 1
                spikeIndex += 1
    return spikeTrain


def filterSpikesTrain(neuronSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
    "Filter spike train using the bi-exponential kernel"
    spikeTrain = createSpikeTrain(neuronSpikes, timeStep, simTimeIni, simTimeFin)
    filteredSignal = np.convolve(spikeTrain, biExponentialKernel(timeStep, timeBoundKernel), 'same')
    return filteredSignal


def calculateCorrCoef(allSpikes, neuronIndex1, neuronIndex2, autoCov, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
    "Calculate the correlation coeficient between spikeTrain1 and spikeTrain2"

    if neuronIndex1 == neuronIndex2:
        return 1
    else:
        Fi = filterSpikesTrain(allSpikes[neuronIndex1], timeStep, simTimeIni, simTimeFin, timeBoundKernel)
        Fj = filterSpikesTrain(allSpikes[neuronIndex2], timeStep, simTimeIni, simTimeFin, timeBoundKernel)
        Vij = np.sum(Fi * Fj)
        Vii = autoCov[neuronIndex1]
        Vjj = autoCov[neuronIndex2]
        #print "Fi: %f\t Fj: %f\t Vij: %f\t Vii: %f\t Vjj: %f" %(np.mean(Fi), np.mean(Fj), Vij, Vii, Vjj)
        return Vij / np.sqrt(Vii * Vjj)


def create_auto_cov(spike_trains, time_step, sim_time_ini, sim_time_fin, time_bound_kernel):  # to speed up processing

    cov = np.zeros(len(spike_trains))
    index = 0
    for spike_train in spike_trains:
        if np.size(spike_train) > 0:
            V = filterSpikesTrain(spike_train, time_step, sim_time_ini, sim_time_fin, time_bound_kernel)
            cov[index] = np.sum(V * V)
        else:
            cov[index] = -1  # To symbolize that there is no spike in this neuron
        index += 1
    return cov


def plotCorrHist(axis, numNeuronsPop, popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel, barColor):
    seg = popSpikes.segments[0]
    allSpikes = seg.spiketrains
    numSpikingNeurons = 0
    indexesSpikingNeurons = np.zeros(numNeuronsPop)

    autoCov = create_auto_cov(allSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel)

    corrCoefs = np.zeros(0)
    for k in range(0, numNeuronsPop):  # Considering only spiking neurons
        #print "k: %d" %k
        if np.size(allSpikes[k]) > 0:
            numSpikingNeurons += 1
            indexesSpikingNeurons[k] = 1

    for i in range(0, numNeuronsPop):
        for j in range(i, numNeuronsPop):
            if indexesSpikingNeurons[i] == 1 and indexesSpikingNeurons[j] == 1 and i != j:
                corrCoef = calculateCorrCoef(allSpikes, i, j, autoCov, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
                corrCoefs = np.append(corrCoefs, corrCoef)
                print "i: %d\tj: %d\tcorrCoef: %f" % (i, j, corrCoef)

    if np.size(corrCoefs) != 0:
        plt.hist(corrCoefs, histtype='stepfilled', color=barColor, alpha=0.7)
    #plt.ylabel('Percent [%]')
    plt.xlabel('Spiking Correlation')
    #plt.ylim((0, 100))
    plt.xlim((0.0, 1.0))
    #axis.set_frame_on(False)
    axis.spines['top'].set_color('none')
    axis.spines['left'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.tick_params(axis='x', top='off')
    axis.tick_params(axis='x', bottom='off')
    axis.tick_params(axis='y', left='off')
    axis.tick_params(axis='y', right='off')
    axis.spines['bottom'].set_linewidth(2)
    axis.spines['left'].set_linewidth(2)


def plot_cor_hist(bar_color, sim_time_fin, sim_time_ini, spike_trains, time_bound_kernel, time_step, print_tag):
    num_spiking_neurons = 0
    num_neurons_pop = len(spike_trains)
    indexes_spiking_neurons = np.zeros(num_neurons_pop)
    auto_cov = create_auto_cov(spike_trains, time_step, sim_time_ini, sim_time_fin, time_bound_kernel)
    corr_coefs = np.zeros(0)
    for k in range(0, num_neurons_pop):  # Considering only spiking neurons
        if np.size(spike_trains[k]) > 0:
            num_spiking_neurons += 1
            indexes_spiking_neurons[k] = 1
    print("\n")
    for i in range(0, num_neurons_pop):
        for j in range(i, num_neurons_pop):
            if indexes_spiking_neurons[i] == 1 and indexes_spiking_neurons[j] == 1 and i != j:
                corr_coef = calculateCorrCoef(spike_trains, i, j, auto_cov, time_step, sim_time_ini, sim_time_fin, time_bound_kernel)
                corr_coefs = np.append(corr_coefs, corr_coef)
                print print_tag + ": i: %d\tj: %d\tcorr_coef: %f" % (i, j, corr_coef)
    if np.size(corr_coefs) != 0:
        corr_coefs = corr_coefs[~np.isnan(corr_coefs)]
        if np.size(corr_coefs) > 1:
            plt.hist(corr_coefs, histtype='stepfilled', color=bar_color, alpha=0.7)


def plot_cor_hist2(bar_color, num_neurons_pop, sim_time_fin, sim_time_ini, spike_trains, time_bound_kernel, time_step):
    num_spiking_neurons = 0
    indexes_spiking_neurons = np.zeros(num_neurons_pop)
    auto_cov = create_auto_cov(spike_trains, time_step, sim_time_ini, sim_time_fin, time_bound_kernel)
    corr_coefs = np.zeros(0)
    for k in range(0, num_neurons_pop):  # Considering only spiking neurons
        if np.size(spike_trains[k]) > 0:
            num_spiking_neurons += 1
            indexes_spiking_neurons[k] = 1
    print("\n")
    for i in range(0, num_neurons_pop):
        for j in range(i, num_neurons_pop):
            if indexes_spiking_neurons[i] == 1 and indexes_spiking_neurons[j] == 1 and i != j:
                corr_coef = calculateCorrCoef(spike_trains, i, j, auto_cov, time_step, sim_time_ini, sim_time_fin, time_bound_kernel)
                corr_coefs = np.append(corr_coefs, corr_coef)
                # print "control: i: %d\tj: %d\tcorr_coef: %f" % (i, j, corr_coef)
    if np.size(corr_coefs) != 0:
        corr_coefs = corr_coefs[~np.isnan(corr_coefs)]
        if np.size(corr_coefs) > 1:
            plt.hist(corr_coefs, histtype='stepfilled', color=bar_color, alpha=0.7)


def plot_corr_double_hist(axis, spike_trains, bar_color, spike_trains2, bar_color2, time_step, sim_time_ini, sim_time_fin, time_bound_kernel):
    plot_cor_hist(bar_color2, sim_time_fin, sim_time_ini, spike_trains2, time_bound_kernel, time_step, "control")
    plot_cor_hist(bar_color, sim_time_fin, sim_time_ini, spike_trains, time_bound_kernel, time_step, "pattern1")

    plt.xlabel('Spiking Correlation')
    plt.xlim((0.0, 1.0))
    axis.spines['top'].set_color('none')
    axis.spines['left'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.tick_params(axis='x', top='off')
    axis.tick_params(axis='x', bottom='off')
    axis.tick_params(axis='y', left='off')
    axis.tick_params(axis='y', right='off')
    axis.spines['bottom'].set_linewidth(2)
    axis.spines['left'].set_linewidth(2)


def isInSubGrid(x, y, xIni, xFin, yIni, yFin):
    "Checks if an element with coordenates (x, y) in a grid is within a sub-grid with the specified limits"
    return (x >= xIni) and (x <= xFin) and (y >= yIni) and (y <= yFin)


def get_spike_train(spikes):
    return spikes.segments[0].spiketrains


def plot_grid(axis, excitatory_spike_trains, pattern1_spike_trains, pattern1_stim_spike_trains, pattern2_spike_trains, pattern2_stim_spike_trains, intersection_spike_trains,
              control_spike_trains, inhib_spike_trains):
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    aux_index_inhib = 0
    aux_index_excitatory = 0
    aux_index_control = 0
    aux_index_pattern1 = 0
    aux_index_pattern1_stim = 0
    aux_index_pattern2 = 0
    aux_index_pattern2_stim = 0
    aux_index_pattern_intersection = 0

    xIniInhib = 0
    xFinInhib = 99
    yIniInhib = 80
    yFinInhib = 99

    xIniControl = 7
    xFinControl = 34
    yIniControl = 11
    yFinControl = 38

    xIniPattern1 = 42
    xFinPattern1 = 69
    yIniPattern1 = 30
    yFinPattern1 = 57

    xIniPattern1_stim = 56
    xFinPattern1_stim = 69
    yIniPattern1_stim = 30
    yFinPattern1_stim = 43

    xIniPattern2 = 22
    xFinPattern2 = 49
    yIniPattern2 = 50
    yFinPattern2 = 77

    xIniPattern2_stim = 22
    xFinPattern2_stim = 35
    yIniPattern2_stim = 64
    yFinPattern2_stim = 77

    xIniPatternIntersection = 42
    xFinPatternIntersection = 49
    yIniPatternIntersection = 50
    yFinPatternIntersection = 57

    grid = np.zeros((100, 100))

    for x in range(100):

        for y in range(100):

            if isInSubGrid(x, y, xIniInhib, xFinInhib, yIniInhib, yFinInhib):
                neuron_spike_times = inhib_spike_trains[aux_index_inhib]
                grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)
                aux_index_inhib += 1

            elif isInSubGrid(x, y, xIniControl, xFinControl, yIniControl, yFinControl):
                neuron_spike_times = control_spike_trains[aux_index_control]
                grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)
                aux_index_control += 1

            elif isInSubGrid(x, y, xIniPattern1, xFinPattern1, yIniPattern1, yFinPattern1):

                if isInSubGrid(x, y, xIniPatternIntersection, xFinPatternIntersection, yIniPatternIntersection, yFinPatternIntersection):
                    neuron_spike_times = intersection_spike_trains[aux_index_pattern_intersection]
                    grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)
                    aux_index_pattern_intersection += 1

                elif isInSubGrid(x, y, xIniPattern1_stim, xFinPattern1_stim, yIniPattern1_stim, yFinPattern1_stim):
                    neuron_spike_times = pattern1_stim_spike_trains[aux_index_pattern1_stim]
                    grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)

                    aux_index_pattern1_stim += 1
                else:
                    neuron_spike_times = pattern1_spike_trains[aux_index_pattern1]
                    grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)
                    aux_index_pattern1 += 1

            elif isInSubGrid(x, y, xIniPattern2, xFinPattern2, yIniPattern2, yFinPattern2):

                if isInSubGrid(x, y, xIniPattern2_stim, xFinPattern2_stim, yIniPattern2_stim, yFinPattern2_stim):
                    neuron_spike_times = pattern2_stim_spike_trains[aux_index_pattern2_stim]
                    grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)
                    aux_index_pattern2_stim += 1

                else:
                    neuron_spike_times = pattern2_spike_trains[aux_index_pattern2]
                    grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times)
                    aux_index_pattern2 += 1

            else:
                neuron_spike_times = excitatory_spike_trains[aux_index_excitatory]
                grid[x, y] = calculate_neuron_firing_rate(neuron_spike_times, )
                aux_index_excitatory += 1

    im = plt.imshow(grid, vmin=0, vmax=200, interpolation='none', cmap=plt.cm.YlOrBr_r)
    return im


def plot_fig4_column(fig, column, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
                   excitatory_spikes, pattern1_spikes, pattern1_stim_spikes, pattern2_spikes, pattern2_stim_spikes, intersection_spikes, control_spikes, inhib_spikes,
                   sampled_pattern1_spikes, sampled_control_spikes):
    ax1 = fig.add_subplot(4, 6, column)

    if column == 1:
        ax1.set_title('pre')
    elif column == 2:
        ax1.set_title('A')
    elif column == 3:
        ax1.set_title('B')
    elif column == 4:
        ax1.set_title('C')
    elif column == 5:
        ax1.set_title('D')
    elif column == 6:
        ax1.set_title('E')

    excitatory_spike_trains = get_spike_train(excitatory_spikes)
    inhib_spike_trains = get_spike_train(inhib_spikes)
    control_spike_trains = get_spike_train(control_spikes)
    intersection_spike_trains = get_spike_train(intersection_spikes)
    pattern1_spike_trains = get_spike_train(pattern1_spikes)
    pattern1_stim_spike_trains = get_spike_train(pattern1_stim_spikes)
    pattern2_spike_trains = get_spike_train(pattern2_spikes)
    pattern2_stim_spike_trains = get_spike_train(pattern2_stim_spikes)

    im = plot_grid(ax1, excitatory_spike_trains, pattern1_spike_trains, pattern1_stim_spike_trains, pattern2_spike_trains, pattern2_stim_spike_trains, intersection_spike_trains,
                   control_spike_trains, inhib_spike_trains)

    ax2 = fig.add_subplot(4, 6, column + 6)
    if column == 1:
        plt.ylabel('Cell no.')
        ax2.spines['left'].set_color('black')
    else:
        ax2.get_yaxis().set_visible(False)

    sampled_pattern1_spike_trains = get_spike_train(sampled_pattern1_spikes)
    sampled_control_spike_trains = get_spike_train(sampled_control_spikes)

    plt.ylim((-15, 15))
    plt.xlim((simTimeIni, simTimeIni + 150))
    plt.xticks(rotation=45)
    plot_raster(ax2, sampled_pattern1_spike_trains, 'red', 1)
    plot_raster(ax2, sampled_control_spike_trains, 'black', -1)
    ax2.tick_params(axis='y', left='on')

    ax3 = fig.add_subplot(4, 6, column + 12)
    if column == 1:
        plt.ylabel('Counts')
        ax3.spines['left'].set_color('black')
    else:
        ax3.get_yaxis().set_visible(False)

    plot_isicv_double_hist(ax3, pattern1_spike_trains, 'red', control_spike_trains, 'black')
    ax3.tick_params(axis='y', left='on')
    ax3.spines['left'].set_color('black')

    ax4 = fig.add_subplot(4, 6, column + 18)
    if column == 1:
        plt.ylabel('Counts')
        ax4.spines['left'].set_color('black')
    else:
        ax4.get_yaxis().set_visible(False)

    plot_corr_double_hist(ax4, sampled_pattern1_spike_trains, 'red', sampled_control_spike_trains, 'black', timeStep, simTimeIni, simTimeFin, timeBoundKernel)

    ax4.spines['left'].set_color('black')
    ax4.tick_params(axis='y', left='on')
    plt.xlim((-0.2, 1.0))

    return im