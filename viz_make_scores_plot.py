import matplotlib.patches as patches
import pandas as pd
import matplotlib.pyplot as plt

"""
@TODO: Add AdaBoost model to complete sklearn models so that it can be judged fairly next to XGBoost.
@TODO: Add LinearSVR too since it is one of the AutoML resulted models?
Not feeling like adding it is not a good reason to avoid it.
"""

# Print font options
if False:
    from matplotlib import font_manager
    font_names = sorted(font_manager.get_font_names())
    print(f"font names {font_names}")

# Ready the data
scores = pd.read_csv("scores.csv")
scores.sort_values(by='rmse', ascending=False, inplace=True)
rmse = scores.rmse / 1000

fig, ax = plt.subplots()
fig.set_dpi(300)
fig.set_size_inches(w=7, h=5)
size_w, size_h = fig.get_size_inches()

print(f"size of pic after setting: {size_w} x {size_h}")

x_labels = "Single Tree", "KNN", "Bagged Trees", "Regularized Linear", "Neural Net", "Random Forest", "XGBoosted Tree", "AutoML"

# Only worry about fiddling with these 3
font_param_axes = {'fontname': 'Liberation Serif', 'fontsize': 13}
font_param_title = {'fontname': 'Liberation Serif', 'fontsize': 15}
font_param_barlabels = {'fontname': 'Liberation Serif', 'fontsize': 12}

# No need to fiddle
font_param_yticks = font_param_axes
font_param_yticks['fontsize'] = font_param_axes['fontsize']-2

# Bar colors
barlist = plt.bar(x_labels, rmse, color='blue')
barlist[6].set_color('orange')
barlist[7].set_color('orange')

# Label axes and title
plt.xticks(ticks=ax.get_xticks(), labels=[])
plt.xlabel('Model', **font_param_axes)
plt.ylabel('RMSE (thousands of dollars)', **font_param_axes)
plt.yticks(ticks=ax.get_yticks(),
           labels=ax.get_yticklabels(), **font_param_yticks)
plt.title("Housing Price Prediction Error", **font_param_title)


def add_bar_labels():
    """ Label bars """
    text_x_offset = 0.12
    y_text = 3
    x_text_range = range(len(x_labels))

    for x, lab in zip(ax.get_xticks(), x_labels):
        ax.text(x=x-text_x_offset, y=y_text, s=lab, rotation=90,
                color='white', **font_param_barlabels)


add_bar_labels()

# Set up legend
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
plt.rcParams['legend.numpoints'] = 1
rect1 = patches.Rectangle((0, 0), 1, 1, facecolor='blue')
rect2 = patches.Rectangle((0, 0), 1, 1, facecolor='orange')
plt.legend((rect1, rect2), ('sklearn', 'companion lib'))

plt.savefig(fname="./article-sklearn-companions/figures/score_barplot.png")
plt.show()
