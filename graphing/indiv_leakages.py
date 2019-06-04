import math
import matplotlib.pyplot as plt
import dill

plt.rcParams["font.family"] = "serif"

category_indices = [('Pkt. Count', (1, 13)),
                    ('Time', (14, 37)),
                    ('Ngram', (38, 161)),
                    ('Transposition', (162, 765)),
                    ('Interval-I', (766, 1365)),
                    ('Interval-II', (1366, 1967)),
                    ('Interval-III', (1968, 2553)),
                    ('Pkt. Distribution', (2554, 2778)),
                    ('Burst', (2779, 2789)),
                    ('First 20', (2790, 2809)),
                    ('First 30', (2810, 2811)),
                    ('Last 30', (2812, 2813)),
                    ('Pkt. per Second', (2814, 2939)),
                    ('CUMUL', (2940, 3043))]


leakage_files = [('undef_indiv.pkl', 'Undefended (FD)', 'grey'),
                 ('wt_indiv.pkl', 'Walkie-Talkie', 'orange'),
                 ('wtfpad_indiv.pkl', 'WTF-PAD', 'cyan'),
                 ('adv_indiv.pkl', 'Mockingbird (FD)', 'red')]

rows = 3
cols = math.ceil(float(len(category_indices)) / rows)

leakages = []
for path, _, __ in leakage_files:
    with open(path, 'r') as fi:
        leakages.append(dill.load(fi))


zipped_leakages = list(zip(*leakages))


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.3)
for i in range(1, len(category_indices)+1):

    category, indices = category_indices[i-1]

    ax = fig.add_subplot(rows, cols, i)
    ax.set_ylim(0, 4)
    ax.set_xticks([indices[0], indices[1]])
    ax.set_yticks(range(0, 5))
    ax.yaxis.grid(True, linestyle='dotted')
    ax.set_title(category, fontsize=11)
    #ax.text(0.5, 0.5, str((rows, cols, i)),
    #       fontsize=18, ha='center')

    for j in range(len(leakage_files)):
        if (j == 1 or j == 3) and (i == 2 or i == 13):
            continue
        x = range(indices[0], indices[1]+1)
        slice = zipped_leakages[indices[0]-1: indices[1]]
        unzip = zip(*slice)
        y = unzip[j]
        ax.plot(x, y, color=leakage_files[j][2])

fig.text(0.085, 0.5, 'Information Leakage (bit)', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.5, 0.03, 'Feature Category', ha='center', va='center', fontsize=16, fontweight='bold')

plt.figlegend(labels=list(zip(*leakage_files))[1], loc='center', ncol=1,
              fontsize=12, bbox_to_anchor=(4.2/cols, 0.65/rows))
plt.show()
