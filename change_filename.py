import os

PATH1 = './attention_map'
PATH2 = 'depth-1_leaf-01'

# model_dir_list = os.listdir(PATH1)
# for model_dir in model_dir_list:
#     if os.path.exists(f'{PATH1}/{model_dir}/{PATH2}'):
#         dir_list = os.listdir(f'{PATH1}/{model_dir}/{PATH2}')
#         for dir in dir_list:
#             dir_path = f'{PATH1}/{model_dir}/{PATH2}/{dir}'
#             if '40x_0.001' not in dir_path:
#                 replace = dir_path.replace('40x','40x_0.001')
#                 os.rename(dir_path,replace)
#                 print(dir_path)
#                 print(replace)
#                 dir_path = replace
            
#     elif os.path.isdir(f'{PATH1}/{model_dir}'):
#         dir_list = os.listdir(f'{PATH1}/{model_dir}')
#         for dir in dir_list:
#             dir_path = f'{PATH1}/{model_dir}/{dir}'
#             if '40x_0.001' not in dir_path:
#                 replace = dir_path.replace('40x','40x_0.001')
#                 os.rename(dir_path,replace)
#                 print(dir_path)
#                 print(replace)
#                 print()
#                 dir_path = replace

model_dir_list = os.listdir(PATH1)
for model_dir in model_dir_list:
    if os.path.exists(f'{PATH1}/{model_dir}/{PATH2}/40x'):
        os.rename(f'{PATH1}/{model_dir}/{PATH2}/40x', f'{PATH1}/{model_dir}/{PATH2}/40x_0.001')
    
    elif os.path.exists(f'{PATH1}/{model_dir}/{PATH2}'):
        if len(os.listdir(f'{PATH1}/{model_dir}/{PATH2}'))>=4:
            file_list = os.listdir(f'{PATH1}/{model_dir}/{PATH2}')
            os.mkdir(f'{PATH1}/{model_dir}/{PATH2}/40x_0.001')
            for file in file_list:
                os.rename(f'{PATH1}/{model_dir}/{PATH2}/{file}', f'{PATH1}/{model_dir}/{PATH2}/40x_0.001/{file}')
    
    if os.path.exists(f'{PATH1}/{model_dir}/40x'):
        os.rename(f'{PATH1}/{model_dir}/40x', f'{PATH1}/{model_dir}/40x_0.001')
    
    elif os.path.exists(f'{PATH1}/{model_dir}'):
        if len(os.listdir(f'{PATH1}/{model_dir}'))>=4:
            file_list = os.listdir(f'{PATH1}/{model_dir}')
            os.mkdir(f'{PATH1}/{model_dir}/40x_0.001')
            for file in file_list:
                os.rename(f'{PATH1}/{model_dir}/{file}', f'{PATH1}/{model_dir}/40x_0.001/{file}')

    