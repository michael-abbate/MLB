'''
pitcher_fct.py has a function, plotPitch(), that plots a strikezone with the pitches given
'''

import matplotlib.pyplot as plt
import seaborn as sns

def plotPitch(px, pz):
    sns.scatterplot(x=px, y = pz)
    count = 0
    for x,y in zip(px,pz):
        labelx, labely = "{:.2f}".format(x), "{:.2f}".format(y)
        plt.annotate((str(count)+': ('+labelx + ', '+ labely+')'), # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0, 4), # distance from text to points (x,y)
                 ha='center',
                 fontsize = 8)
        count += 1

    plt.ylim(0,5)
    plt.xlim(-5,5)
    plt.plot([-1, -1], [1.5, 3.5], linewidth=1, color='grey', linestyle='dashed')
    plt.plot([-1, 1], [1.5, 1.5], linewidth=1, color='grey', linestyle='dashed')
    plt.plot([1, 1], [1.5, 3.5], linewidth=1, color='grey', linestyle='dashed')
    plt.plot([-1, 1], [3.5, 3.5], linewidth=1, color='grey', linestyle='dashed')
    plt.show()

