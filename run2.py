import subprocess
import sys

command_list = [
    ['-c','subtype'],
    ['-c','leaf'], 
    ['-c','new_tree'], 
    ['-c','new_tree','-a'],
    ['-c','new_tree','-l','myinvarse'],
    ['-c','new_tree','-l','LDAM','-C','0.1'],
    ['-c','new_tree','-l','LDAM','-C','0.3'],
    ['-c','new_tree','-l','LDAM','-C','0.5'],
    ['-c','new_tree','-l','LDAM','-C','0.2'],
    ['-c','new_tree','-l','myinvarse','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.1','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.2','--fc'],
    ['-c','new_tree','--fc'], 
]

split_list = [
    ['123','5'],
    ['234','1'],
    ['345','2'],
    ['451','3'],
    ['512','4']
]

args = sys.argv


mode_option = [[0,1,2],[3,4,5],[6,7,8],[9,10,11,12]]
mode_list = mode_option[int(args[1])]
# mode_list = range(len(command_list))
gpu = int(args[1])

for mode in mode_list:
    for split in split_list:
        command1 = ['python','MIL_test.py']+split+['--depth','1','--leaf','01']
        command2 = ['--gpu',f'{gpu}']+command_list[mode]
        command = command1 + command2
        print(command)
        subprocess.run(command)
    
    command = ['python','make_log_Graphs.py','--depth','1','--leaf','01']+command_list[mode]
    subprocess.run(command)

    command = ['python','draw_heatmap.py','--depth','1','--leaf','01']+command_list[mode]
    print(command)
    subprocess.run(command)

