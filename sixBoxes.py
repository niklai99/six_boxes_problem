################################################################################
#                                                                              #
#                       Written by NikLai, 15/04/2021                          #
#                                                                              #
################################################################################

################################    USAGE     ##################################
#                                                                              #
#                   python sixBoxes.py nSamplings nWhites                      #
#                        python sixBoxes.py nSamplings                         #
#                                                                              #
#                nSamplings = number of samplings to perform                   #
#                nWhites = number of whites in the chosen box                  #
#                 NOTE: if nWhites is not provided as argument                 #
#                       then the box is chosen randomly                        #
#                                                                              #
################################################################################


from sys import argv, exit
import numpy as np
from numpy.random import binomial, randint
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(89540)

# Box class
class Box:

    # init method
    def __init__(self, W):
        # number of whites in the box
        self.whites = W


# Sampling class
class Sample(Box):

    # init method
    def __init__(self, W, n):
        # number of whites in the sampling box
        super().__init__(W)
        # number of sampling iterations
        self.iterations = n
        # sampling probabilities for each box
        self.prob = np.array([w/5 for w in range(5+1)])
        # inferring probability of the boxes
        self.P = np.empty((n+1, 6))
        # outcomes from the sampling loop
        self.outcome = np.empty(n)
        # sum of all positive samplings
        self.sum = 0
        # probability of extracting white next sample
        self.wProb = np.empty(n+1)
        


    # infer probabilites method
    def computeProb(self):

        # initialize first probabilities as flat
        self.P[0] = 1/6
        self.wProb[0] = 1/2

        # initialize no outcome i.e. first flat probabilty before sampling
        # self.outcome[0] = 0
        # sampling loop
        for i in range(1, self.iterations+1):

            # sample from the chosen box
            S = binomial(n=1, size=1, p=self.prob[self.whites])

            # save the outcome
            self.outcome[i-1] = int(S)

            # compute total number of whites sampled
            self.sum += S

            # compute probability for each box
            self.P[i] = self.prob**self.sum * (1-self.prob)**(i-self.sum) / np.sum(self.prob**self.sum * (1-self.prob)**(i-self.sum))
            
            # probability of extracting white next sample
            self.wProb[i] = np.sum(self.prob * self.P[i])
            



    def plotProb(self):

        # initialize number of points to plot
        # number of sampling iterations + the probability before the first extraction
        n = self.iterations + 1

        # generate the x-axis grid
        grid = range(n)

        # figure and axes
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14,7), sharex=True, sharey = True, squeeze=False)

        # figure suptitle
        fig.suptitle('The "Six Boxes" Problem', fontsize = 24)

        # axes limits
        ax[0][0].set_xlim(left = -1, right = n)
        ax[0][0].set_ylim(bottom = -0.1, top = 1.1)

        # axis labels
        ax[1][0].set_xlabel('# extraction', fontsize = 18)
        ax[1][1].set_xlabel('# extraction', fontsize = 18)
        ax[1][2].set_xlabel('# extraction', fontsize = 18)
        ax[0][0].set_ylabel('probability', fontsize = 18)
        ax[1][0].set_ylabel('probability', fontsize = 18)


        # dictionary to map outcomes as color:
        # if the result of the extraction is 0 (black stone) then the point will be plotted as orange
        # if the result of the extraction is 1 (white stone) then the point will be plotted as blue
        colors = {1:'#006FFF', 0:'#FF6300'}

        # loop over axes
        h = 0
        for j in range(2):
            for k in range(3):

                # set horizontal grid lines
                ax[j][k].set_axisbelow(True)
                ax[j][k].yaxis.grid(color='black', linestyle='dashed', alpha = 0.2)

                # plot the probability before any extraction as a black point
                ax[j][k].scatter(grid[0], self.P[0, j+k+h], marker = '.', color='black', s = 80)

                # plot the probability after each extraction 
                sns.scatterplot(x=grid[1:], y=self.P[1:, j+k+h], hue=self.outcome, palette=colors, s = 40, ax = ax[j][k])

                # set axes title
                ax[j][k].set_title('Box ' + format(j+k+h, '1.0f'), fontsize = 20)

                # set axes ticks
                ax[j][k].tick_params(axis = 'both', which = 'major', labelsize = 16, direction = 'out', length = 5)

                # make custom legend labels
                handles, labels  =  ax[j][k].get_legend_handles_labels()
                ax[j][k].legend(np.flip(handles), ['White', 'Black'], loc='best')

            h+=2

        fig.tight_layout()
        

    
    def plotWhite(self):

        # initialize number of points to plot
        # number of sampling iterations + the probability before the first extraction
        n = self.iterations + 1

        # generate the x-axis grid
        grid = range(n)

        # figure and axes
        fig, ax = plt.subplots(figsize=(5.5,3.5))

        # axes limits
        ax.set_xlim(left = -1, right = n)
        ax.set_ylim(bottom = -0.1, top = 1.1)

        # axis labels
        ax.set_xlabel('# extraction', fontsize = 18)
        ax.set_ylabel('probability', fontsize = 18)

        # set horizontal grid lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='black', linestyle='dashed', alpha = 0.2)

        # set axes ticks    
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16, direction = 'out', length = 5)

        # set axes title
        ax.set_title('White Sampling Probability', fontsize = 20)

        # dictionary to map outcomes as color:
        # if the result of the extraction is 0 (black stone) then the point will be plotted as orange
        # if the result of the extraction is 1 (white stone) then the point will be plotted as blue
        colors = {1:'#006FFF', 0:'#FF6300'}

        # plot the probability before any extraction as a black point
        ax.scatter(grid[0], self.wProb[0], marker = '.', color='black', s = 80)

        # plot the probability after each extraction 

        # USING MATPLOTLIB = NO LEGEND
        # the color of the point relates to the i-th extraction
        # the value of the point is the probability of extracting white THE FOLLOWING (i+1)-th EXTRACTION!
        # ax.scatter(grid[1:], self.wProb[1:], marker = '.', c=list(map(lambda x: colors[x], self.outcome)), s = 80)

        # USING SEABORN
        sns.scatterplot(x=grid[1:], y=self.wProb[1:], hue=self.outcome, palette=colors, s = 40, ax = ax)

        # make custom legend labels
        handles, labels  =  ax.get_legend_handles_labels()
        ax.legend(np.flip(handles), ['White', 'Black'], loc='best')

        fig.tight_layout()
        


def main(argv):

    
    if len(argv) > 2:
        print('Too many arguments!')
        exit(1)
    else:

        try:
            # number of iterations
            n = int(argv[0])
        except IndexError:
            print('Missing "number of iterations" argument')
            exit(1)
        except ValueError:
            print('Argument "number of iterations" must be integer')
            exit(1)

        try:
            # chosen box
            B = int(argv[1]) 
        except ValueError:
            print('Argument "sampling box" must be integer between 0 and 5')
            exit(1)
        except IndexError:
            print('No "sampling box" argument provided: a random box will be chosen')
            # sample random box
            B = randint(0,6)
        else:
            if B not in range(6):
                print('Argument "sampling box" must be integer between 0 and 5')
                exit(1)
            
 
    # sample n times from box B 
    sample = Sample(B, n)
    
    # compute probabilities for each of the six boxes
    sample.computeProb()


    # plot probabilities 
    sample.plotProb()

    # plot white sample probabilities
    sample.plotWhite()

    plt.show()

    return


if __name__ == "__main__":
   main(argv[1:])