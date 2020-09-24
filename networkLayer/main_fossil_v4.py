from networkLayer.configs.fossil_v4 import opt
import networkLayer.trainer_v3 as Trainer
from networkLayer.utils import save_eval_images, check_dir

def inference(sample):
    trainer = Trainer.Trainer_Basic(opt).cuda()
    trainer.set_input(sample)
    trainer.inference()
    trainer.get_current_errors()
    visuals = trainer.get_current_visuals()

    if True:
        fossil_name = sample['name'][0].split('/')[-3]
        save_root = opt.logger.log_dir + '/' + fossil_name

        img_dir = save_root  # '../checkpoints/log/'
        check_dir(img_dir)
        save_eval_images(visuals, img_dir, 1, opt)

    return visuals
