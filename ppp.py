import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn')
x = np.arange(0, (6-1)*2.5+1, 2.5)  # the label locations
width = 1.0  # the width of the bars
fig, ax = plt.subplots(figsize=(14.8, 7.8))

subj_train_accs = [1, 1, 1, 1, 1, 1]
subj_test_accs = [0.5323, 0.5323, 0.5323, 0.5323, 0.5323, 0.5323]
subj_train_f1s = [1, 1, 1, 1, 1, 1]
subj_test_f1s = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
acc_train_rect = ax.bar(x - width/2, subj_train_accs, width, label='Train/Acc', fill=False, ls='--')
acc_test_rect = ax.bar(x - width/2, subj_test_accs, width, label='Test/Acc')
f1_train_rect = ax.bar(x + width/2, subj_train_f1s, width, label='Train/F1', fill=False, ls='--')
f1_test_rect = ax.bar(x + width/2, subj_test_f1s, width, label='Test/F1')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Subjects')
ax.set_title('sdnaskjduendjand', pad=36)
ax.set_xticks(x)
ax.set_xticklabels([1, 2, 3, 4, 5, 6])
ax.set_ylim(0.0, 1.0)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend([acc_train_rect, acc_test_rect, f1_test_rect], ['Train', 'Test/Acc.', 'Test/F1.'], loc='center left', bbox_to_anchor=(1, 0.5))
ax.bar_label(acc_train_rect, padding=3)
ax.bar_label(acc_test_rect, padding=3)
ax.bar_label(f1_train_rect, padding=3)
ax.bar_label(f1_test_rect, padding=3)
fig.savefig('./figs/test.png')
plt.close('all')