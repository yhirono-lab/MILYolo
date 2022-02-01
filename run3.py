import subprocess
import sys

# 7/29時点の全コマンドリスト
command_list = [
    ['--depth','1','--leaf','01','-c','subtype','-l','normal'],
    ['--depth','1','--leaf','01','-c','subtype','-l','normal','--fc'],
    ['--depth','1','--leaf','01','-c','leaf','-l','normal'],
    ['--depth','1','--leaf','01','-c','leaf','-l','normal','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','normal'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','myinvarse'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.1'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.2'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.3'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','normal','-a'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','myinvarse','-a'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.1','-a'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.2','-a'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.3','-a'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','-a'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','normal','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','myinvarse','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.1','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.2','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.3','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--fc'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','normal','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','myinvarse','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.1','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.3','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','normal','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','myinvarse','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.1','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.3','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','myinvarse','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.1','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.3','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','1.0','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','2.0','--data','add'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','1.0','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','2.0','--data','add','--model','vgg11'],
    # ['--depth','1','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add','--lr','0.0001'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','normal','--data','add'],
    # ['--depth','3','--leaf','01','-c','new_tree','-l','normal','--data','add'],
    # ['--depth','4','--leaf','01','-c','new_tree','-l','normal','--data','add'],
    # ['--depth','2','--leaf','01','-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    # ['--depth','3','--leaf','01','-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    # ['--depth','4','--leaf','01','-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    # ['--depth','2','--leaf','01','-c','new_tree','-l','myinvarse','--data','add'],
    # ['--depth','3','--leaf','01','-c','new_tree','-l','myinvarse','--data','add'],
    # ['--depth','4','--leaf','01','-c','new_tree','-l','myinvarse','--data','add'],
    # ['--depth','2','--leaf','01','-c','new_tree','-l','myinvarse','--data','add','--model','vgg11'],
    # ['--depth','3','--leaf','01','-c','new_tree','-l','myinvarse','--data','add','--model','vgg11'],
    # ['--depth','4','--leaf','01','-c','new_tree','-l','myinvarse','--data','add','--model','vgg11'],
    # ['--depth','2','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add'],
    # ['--depth','3','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add'],
    # ['--depth','4','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add'],
    # ['--depth','2','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add','--model','vgg11'],
    # ['--depth','3','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add','--model','vgg11'],
    # ['--depth','4','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add','--model','vgg11'],
]

split_list = [
    ['123','4','5'],
    ['234','5','1']
]

args = sys.argv
if len(args)!=2:
    exit()

# mode_option = [[1,0],[2,3],[5,4],[6,7,8]]
mode_option = [list(range(len(command_list)))]
mode_list = mode_option[int(args[1])]

gpu = int(args[1])

for mode in mode_list:
    for split in split_list:
    #     command1 = ['python','MIL_train_Single.py']+[split[0], split[1]]+['--depth','1','--leaf','01']
    #     command2 = ['--gpu',f'{gpu}']+command_list[mode]
    #     command = command1 + command2
    #     print(command)
    #     subprocess.run(command)

        command1 = ['python','MIL_test.py']+[split[0], split[2]]
        command2 = ['--gpu',f'{gpu}']+command_list[mode]
        command = command1 + command2
        print(command)
        subprocess.run(command)
    
    command = ['python','make_log_Graphs.py','--depth','1','--leaf','01']+command_list[mode]
    print(command)
    subprocess.run(command)

    command = ['python','draw_heatmap.py','--depth','1','--leaf','01']+command_list[mode]
    print(command)
    subprocess.run(command)

