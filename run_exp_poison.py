from scalablerunner.taskrunner import TaskRunner
from dataset import Backdoor

if __name__ == "__main__":
    config = {
        # 'Poison Rate':{
        #     'TWCC - 0,1': {
        #         'Call': "python diffusers_training_example.py",
        #         'Param': {
        #             '--postfix': ['new-set-1', 'new-set-2'],
        #             '--project': ['Poison_Rates'],
        #             '--mode': ['train+measure'],
        #             '--dataset': ['CIFAR10'],
        #             '--batch': [128],
        #             '--epoch': [50],
        #             '--clean_rate': [1],
        #             '--poison_rate': [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0],
        #             '--trigger': [Backdoor.TRIGGER_BOX_14, Backdoor.TRIGGER_STOP_SIGN_14],
        #             # '--trigger': [Backdoor.TRIGGER_BOX_14],
        #             '--target': [Backdoor.TARGET_TG, Backdoor.TARGET_BOX, Backdoor.TRIGGER_FA, Backdoor.TARGET_FEDORA_HAT, Backdoor.TARGET_SHIFT],
        #             '--ckpt': ['DDPM-CIFAR10-32'], 
        #             '--fclip': ['w'],
        #             '': ['-o']
        #         },
        #         'Async':{
        #             # '--gpu': ['0', '1', '0', '1', '0', '1']
        #             '--gpu': ['0', '1', '2', '3', 
        #                       '0', '1', '2', '3', 
        #                       '0', '1', '2', '3', 
        #                       '0', '1', '2', '3']
        #         }
        #     }, 
        # },
        'Poison Rate':{
            'TWCC - 0,1': {
                'Call': "python baddiffusion.py",
                'Param': {
                    '--postfix': ['new-set-1'],
                    '--project': ['test'],
                    '--mode': ['train+measure'],
                    '--dataset': ['CELEBA-HQ'],
                    # '--batch': [128],
                    '--batch': [64],
                    '--epoch': [1],
                    '--clean_rate': [1],
                    '--poison_rate': [1.0],
                    '--trigger': [Backdoor.TRIGGER_BOX_18, Backdoor.TRIGGER_STOP_SIGN_8],
                    # '--trigger': [Backdoor.TRIGGER_BOX_14],
                    '--target': [Backdoor.TARGET_HAT],
                    # '--ckpt': ['DDPM-CIFAR10-32'], 
                    '--ckpt': ['DDPM-CELEBA-HQ-256'], 
                    '--fclip': ['o'],
                    '': ['-o']
                },
                'Async':{
                    # '--gpu': ['0', '1', '0', '1', '0', '1']
                    '--gpu': ['0', '1', '2', '3',]
                }
            }, 
        },
        # 'Measure CIFAR10':{
        #     'TWCC': {
        #         'Call': "python diffusers_training_example.py",
        #         'Param': {
        #             '--project': ['Poison_Rates'],
        #             '--mode': ['measure'],
        #             # '--dataset': ['CIFAR10'],
        #             '--ckpt': [
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_BOX_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-SHIFT_new-set-1',
                            
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.9_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.7_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.5_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.2_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.05_STOP_SIGN_14-SHIFT_new-set-1',

        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_STOP_SIGN_14-TRIGGER_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_STOP_SIGN_14-BOX_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_STOP_SIGN_14-FASHION_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_STOP_SIGN_14-FEDORA_HAT_new-set-1',
        #                     'res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_STOP_SIGN_14-SHIFT_new-set-1',
        #                     ], 
        #             '--fclip': ['w'],
        #         },
        #         'Async':{
        #             '--gpu': ['0', '1']
        #         }
        #     }, 
        # }   
    }
    
    tr = TaskRunner(config=config)
    tr.run()
