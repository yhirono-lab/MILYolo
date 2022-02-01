import subprocess
import sys

command_list = [
    ['--depth','1','--leaf','01','--lr', '0.0005', '--dropout', '--data', 'add', '--reduce','--multistage','3','--detect_obj','50'],
    ['--depth','2','--leaf','01','--lr', '0.0005', '--dropout', '--data', 'add', '--reduce','--multistage','2','--detect_obj','50'],
    ['--depth','1','--leaf','01','--lr', '0.0005', '--dropout', '--data', 'add', '--reduce','--multistage','3','--detect_obj','20'],
    ['--depth','2','--leaf','01','--lr', '0.0005', '--dropout', '--data', 'add', '--reduce','--multistage','2','--detect_obj','20'],
]

split_list = [
    ['123'],
    ['234'],
]

mode_list = range(len(command_list))

for mode in mode_list:
    for split in split_list:
        command1 = ['python','train_new.py']+[split[0]]
        command2 = command_list[mode]
        command = command1 + command2
            
        print(command)
        subprocess.run(command)

        command1 = ['python','test_new.py']+[split[0]]
        command2 = command_list[mode]+['--exist_ok']
        command = command1 + command2
        command = [c for c in command if c != '-r']
        print(command)
        subprocess.run(command)
    
    command = ['python','result_analyse.py']+command_list[mode]
    command = [c for c in command if c != '-r']
    print(command)
    subprocess.run(command)

    command = ['python','draw_heatmap.py']+command_list[mode]
    command = [c for c in command if c != '-r']
    print(command)
    subprocess.run(command)

