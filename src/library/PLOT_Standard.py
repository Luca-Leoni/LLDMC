import matplotlib.pyplot as plt

##############################
#
#   STANDARD PLOTTING FUNCTIONS
#
##############################

#***Cosmetic***
def plotSetUp(large = 22, med = 16, small = 12):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage[italic,eulergreek]{mathastext}')

    # setting paramater style
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': large,
              'ytick.labelsize': med,
              'figure.titlesize': large,
              "pgf.texsystem": "pdflatex",
              'font.family': 'serif',
              'text.usetex': True,
              'pgf.rcfonts': False}
    plt.rcParams.update(params)
    
    # Adjust the margins
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.99)

    # Settinf style for the whole picture
    plt.style.use('classic')

#***Funzione per creare in automatico il plot con le specifiche volute***
def createPlot(Ncol: int, Nrow: int, shareX: bool = False, shareY: bool = False, sizeX = 17, sizeY = 15, TickYsize = 14, TickXsize = 14, Xticks = True, Yticks = True, widthRateo = None):
    if widthRateo == None:
        fig, ax = plt.subplots(figsize=(sizeX,sizeY), ncols=Ncol, nrows=Nrow, sharex=shareX, sharey=shareY)
    else:
        fig, ax = plt.subplots(figsize=(sizeX,sizeY), ncols=Ncol, nrows=Nrow, sharex=shareX, sharey=shareY, gridspec_kw={'width_ratios': widthRateo})



    if Ncol == 1:
        if Nrow == 1:
            ax.grid(True)
            ax.tick_params(axis='x', which='major', labelsize=TickXsize, labelbottom=Xticks)
            ax.tick_params(axis='y', which='major', labelsize=TickYsize, labelleft=Yticks)
            for l in ['top', 'left', 'right', 'bottom']:
                ax.spines[l].set_linewidth(2)
        else:
            for i in range(0, Nrow):
                ax[i].grid(True)
                ax[i].tick_params(axis='x', which='major', labelsize=TickXsize, labelbottom=Xticks)
                ax[i].tick_params(axis='y', which='major', labelsize=TickYsize, labelleft=Yticks)
                for l in ['top', 'left', 'right', 'bottom']:
                    ax[i].spines[l].set_linewidth(2)
    else:
        if Nrow == 1:
            for i in range(0, Ncol):
                ax[i].grid(True)
                ax[i].tick_params(axis='x', which='major', labelsize=TickXsize, labelbottom=Xticks)
                ax[i].tick_params(axis='y', which='major', labelsize=TickYsize, labelleft=Yticks)
                for l in ['top', 'left', 'right', 'bottom']:
                    ax[i].spines[l].set_linewidth(2)
        else:
            for i in range(0,Nrow):
                for j in range(0,Ncol):
                    ax[i][j].grid(True)
                    ax[i][j].tick_params(axis='x', which='major', labelsize=TickXsize, labelbottom=Xticks)
                    ax[i][j].tick_params(axis='y', which='major', labelsize=TickYsize, labelleft=Yticks)
                    for l in ['top', 'left', 'right', 'bottom']:
                        ax[i][j].spines[l].set_linewidth(2)
    
    plotSetUp()
    fig.patch.set_facecolor('white')

    return fig, ax