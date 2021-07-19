import os
import csv

from psychopy.visual import Window, TextBox
from psychopy.gui import DlgFromDict, Dlg
from psychopy.event import waitKeys
from psychopy.core import quit
from psychopy.hardware.keyboard import Keyboard


def wait_for_keyinput():
    while True:
        keys = kb.getKeys()
        if "space" in keys:
            break
        elif "escape" in keys:
            win.close()
            quit()


def start_artwork_exp():
    # Welcome
    welcome_title = TextBox(win, text='油画情绪分类实验', font_size=32, font_color=[-1, -1, -1], pos=[0.0, 0.0], bold=True)
    welcome_text = TextBox(win, text='{}, 你好，感谢你参加本次实验。'.format(subj_info['姓名']), font_size=18, font_color=[-1, -1, -1], pos=[0.0, 0.4])
    instruction_text = TextBox(win, text='按空格键继续', font_size=10, font_color=[0.5, 0.5, 0.5], pos=[0.0, 0.8])
    
    welcome_title.draw()
    welcome_text.draw()
    instruction_text.draw()
    win.flip()

    # Tutorial
    tutorial_title = TextBox(win, text='实验介绍', font_size=32, font_color=[-1, -1, -1], pos=[0.0, -0.8], bold=True)
    tutorial_text = TextBox(win, text='感谢您参与实验')

if __name__ == '__main__':

    pics_path = r'/Users/luoshuai/Pictures/ArtsCollection_v1'
    record_dir = './artwork_exp_record'
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    # Subject information
    subj_info = {'姓名':'', '姓名拼音':'', '性别':['男', '女'], '年龄':'', '国籍':''}
    subj_info_dlg = DlgFromDict(subj_info, title='请填写个人信息', order=['姓名', '姓名拼音', '性别', '年龄', '国籍'])

    if not subj_info_dlg.OK:
        quit()
    else:
        if not subj_info['姓名'] or not subj_info['姓名拼音']:
            quit()
    
    subj_dir = record_dir + '/' + subj_info['姓名拼音']
    os.makedirs(subj_dir)
    record_file = subj_dir + '/' + subj_info['姓名拼音'] + '.csv'
    csvfile = open(record_file, mode='w', encoding='utf-8', newline='')
    csvwriter = csv.writer(csvfile, delimiter=',')
    
    # Start
    win = Window(color=(0.06, 0.06, 0.06), fullscr=True)
    kb = Keyboard()

    start_artwork_exp()
    
    csvfile.close()
    win.close()
    quit()